"""Multilingual term lists for the indices layer (Slice 2).

Each lexicon is a plain Python literal — no external download — so the
indices layer remains Pyodide-clean.

Top-level structure::

    LEXICONS[index_name][language] = frozenset(...) | dict[str, frozenset]

The EPU and LUI lexicons use a nested dict (multi-group co-occurrence
methodology); MPU/GPR/WUI use a flat frozenset; tone uses a
hawkish/dovish dict. LUI specifically uses three keys per language:
``labor_domain``, ``uncertainty_tone``, and ``phrases``.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Baker-Bloom-Davis EPU — three co-occurring term groups (Economy / Policy
# / Uncertainty). A document counts toward EPU if it contains ≥1 term from
# each group.
# ---------------------------------------------------------------------------
_EPU_EN = {
    "economy":     frozenset({"economic", "economy", "economics"}),
    "policy":      frozenset({"policy", "policies", "regulation", "regulatory",
                              "legislation", "deficit", "tariff",
                              "white house", "congress", "senate", "house",
                              "federal reserve", "central bank"}),
    "uncertainty": frozenset({"uncertain", "uncertainty", "uncertainties"}),
}


# ---------------------------------------------------------------------------
# Husted-Rogers-Sun monetary-policy uncertainty — flat term list.
# ---------------------------------------------------------------------------
_MPU_EN = frozenset({
    "monetary", "policy", "policies",
    "federal reserve", "central bank", "fomc", "ecb", "boe", "boj",
    "interest rate", "interest rates", "policy rate",
    "uncertain", "uncertainty", "uncertainties",
    "ambiguity", "ambiguous",
})


# ---------------------------------------------------------------------------
# Caldara-Iacoviello geopolitical-risk index — flat term list.
# ---------------------------------------------------------------------------
_GPR_EN = frozenset({
    "war", "warfare", "military",
    "terror", "terrorism", "terrorist", "terrorists",
    "geopolitical", "geopolitics",
    "sanctions", "sanction",
    "invasion", "invade",
    "nuclear", "missile", "missiles",
    "conflict", "tensions",
})


# ---------------------------------------------------------------------------
# Apel-Blix-Grimaldi hawkish/dovish tone lexicon (English).
# ---------------------------------------------------------------------------
_TONE_EN = {
    "hawkish": frozenset({
        "hawkish", "tighten", "tightening", "tightened",
        "hike", "hiked", "hikes",
        "raise", "raised", "raises",
        "restrictive", "withdraw", "withdrawal",
        "inflationary", "overheating",
    }),
    "dovish": frozenset({
        "dovish", "ease", "eased", "easing",
        "cut", "cuts", "cutting",
        "lower", "lowered", "lowers",
        "accommodative", "accommodation",
        "stimulus", "support",
        "deflationary", "slack",
    }),
}


# ---------------------------------------------------------------------------
# Ahir-Bloom-Furceri World Uncertainty Index — flat term list (uncertainty
# stems only).
# ---------------------------------------------------------------------------
_WUI_EN = frozenset({
    "uncertain", "uncertainty", "uncertainties",
    "ambiguity", "ambiguous",
    "unpredictable", "unpredictability",
    # Slice 6a additions — Hubert-inspired economic-uncertainty vocabulary.
    # Drawn from open peer-reviewed sources on inflation expectations
    # uncertainty (Hubert 2017; Coibion-Gorodnichenko 2015).
    "inflation expectations", "anchored expectations", "unanchored expectations",
    "deviation from target", "off target",
    "second-round effects", "second round effects",
    "wage-price spiral", "price spiral",
    "price stability concerns",
    "policy uncertainty", "policy unpredictability",
    "macroeconomic uncertainty",
    "downside risks", "upside risks",
    "tail risk", "tail risks", "fat tails",
    "stagflation", "stagflationary",
    "headwinds", "crosscurrents",
    "fragile recovery",
    "data-dependent",
    "uncertain outlook", "uncertain path",
    "asymmetric risks",
})


# ---------------------------------------------------------------------------
# Labor-domain vocabulary (Slice 6a) — broad labor-economics terms,
# polarity-neutral. Used as one of two co-occurring groups in the
# sentence-level LUI scoring (the other is _UNCERTAINTY_TONE_<lang>).
# Curated to be substantially broader than _LUI_PHRASES_<lang> (which
# is high-precision pre-formed phrases).
# ---------------------------------------------------------------------------
_LABOR_DOMAIN_EN = frozenset({
    "labor", "labour",
    "labor market", "labour market",
    "labor force", "labour force",
    "employment", "unemployment", "underemployment",
    "jobs", "job", "job market", "job creation", "job openings",
    "hiring", "hire", "hires",
    "wages", "wage", "salaries", "salary", "compensation", "earnings",
    "workforce", "workers", "worker", "employees", "employee",
    "payroll", "payrolls", "nonfarm payrolls",
    "labor cost", "labor costs", "unit labor cost", "unit labor costs",
    "vacancies", "vacancy",
    "layoffs", "layoff",
    "quit rate", "quits",
    "participation", "participation rate",
    "jobless", "jobless rate", "unemployment rate",
    "labor productivity",
    "labor demand", "labor supply",
    "openings",
    "wage growth",
})

_LABOR_DOMAIN_ES = frozenset({
    "trabajo", "mercado laboral", "mercado de trabajo",
    "fuerza laboral", "fuerza de trabajo",
    "empleo", "desempleo", "subempleo",
    "puestos de trabajo", "puesto de trabajo",
    "creación de empleo",
    "contratación", "contrataciones",
    "salarios", "salario", "sueldos", "sueldo", "remuneración",
    "trabajadores", "trabajador", "empleados", "empleado",
    "nómina", "nóminas",
    "costos laborales", "coste laboral",
    "vacantes", "vacante",
    "despidos", "despido",
    "tasa de paro", "tasa de desempleo", "tasa de empleo",
    "tasa de participación",
    "productividad laboral",
    "demanda laboral", "oferta laboral",
    "crecimiento salarial",
})

_LABOR_DOMAIN_PT = frozenset({
    "trabalho", "mercado de trabalho", "mercado laboral",
    "força de trabalho",
    "emprego", "desemprego", "subemprego",
    "vagas de emprego", "vagas",
    "criação de empregos", "criação de vagas",
    "contratação", "contratações",
    "salários", "salário", "remuneração", "ganhos",
    "trabalhadores", "trabalhador", "empregados", "empregado",
    "folha de pagamento", "folha salarial",
    "custos do trabalho", "custo do trabalho",
    "demissões", "demissão",
    "taxa de desemprego", "taxa de emprego",
    "taxa de participação",
    "produtividade do trabalho",
    "demanda por trabalho", "oferta de trabalho",
    "crescimento salarial",
})

_LABOR_DOMAIN_DE = frozenset({
    "arbeit", "arbeitsmarkt",
    "erwerbstätigkeit", "erwerbstätige",
    "beschäftigung", "arbeitslosigkeit", "unterbeschäftigung",
    "arbeitsplätze", "arbeitsplatz",
    "schaffung von arbeitsplätzen",
    "einstellung", "einstellungen",
    "löhne", "lohn", "gehälter", "gehalt", "vergütung",
    "arbeitnehmer", "arbeiter", "beschäftigte",
    "lohnkosten", "arbeitskosten",
    "stellenangebote", "offene stellen",
    "entlassungen", "entlassung",
    "arbeitslosenquote", "beschäftigungsquote",
    "erwerbsquote",
    "arbeitsproduktivität",
    "arbeitsnachfrage", "arbeitsangebot",
    "lohnwachstum",
})

_LABOR_DOMAIN_FR = frozenset({
    "travail", "marché du travail",
    "main-d'œuvre", "force de travail",
    "emploi", "chômage", "sous-emploi",
    "postes", "postes de travail",
    "création d'emplois",
    "embauche", "embauches", "recrutement",
    "salaires", "salaire", "rémunération", "rémunérations",
    "travailleurs", "travailleur", "salariés", "salarié",
    "masse salariale",
    "coûts du travail", "coût du travail",
    "offres d'emploi", "postes vacants",
    "licenciements", "licenciement",
    "taux de chômage", "taux d'emploi",
    "taux d'activité",
    "productivité du travail",
    "demande de travail", "offre de travail",
    "croissance des salaires",
})

_LABOR_DOMAIN_IT = frozenset({
    "lavoro", "mercato del lavoro",
    "forza lavoro", "forza-lavoro",
    "occupazione", "disoccupazione", "sottoccupazione",
    "posti di lavoro", "posto di lavoro",
    "creazione di posti di lavoro",
    "assunzioni", "assunzione",
    "salari", "salario", "stipendi", "stipendio", "retribuzioni",
    "lavoratori", "lavoratore", "occupati", "dipendenti",
    "monte salari",
    "costo del lavoro", "costi del lavoro",
    "posti vacanti", "vacanze",
    "licenziamenti", "licenziamento",
    "tasso di disoccupazione", "tasso di occupazione",
    "tasso di partecipazione",
    "produttività del lavoro",
    "domanda di lavoro", "offerta di lavoro",
    "crescita salariale",
})

_LABOR_DOMAIN_JA = frozenset({
    "労働", "労働市場",
    "労働力",
    "雇用", "失業", "不完全雇用",
    "就業", "就労",
    "雇用創出",
    "採用", "新規採用",
    "賃金", "給料", "給与", "報酬",
    "労働者", "従業員", "被用者",
    "人件費",
    "求人", "求人数", "求人倍率",
    "解雇", "離職",
    "失業率", "雇用率", "労働参加率",
    "労働生産性",
    "労働需要", "労働供給",
    "賃金上昇",
})

_LABOR_DOMAIN_ZH = frozenset({
    "劳动", "劳动力市场", "就业市场",
    "劳动力",
    "就业", "失业", "不充分就业",
    "工作", "岗位", "职位",
    "创造就业",
    "招聘", "录用",
    "工资", "薪资", "薪酬", "报酬",
    "工人", "员工", "雇员",
    "工资总额",
    "招聘需求", "用工需求",
    "解雇", "辞退",
    "失业率", "就业率",
    "劳动参与率",
    "劳动生产率",
    "劳动需求", "劳动供给",
    "工资增长",
})


# ---------------------------------------------------------------------------
# Uncertainty/risk tone markers (Slice 6a) — polarity-neutral language
# that signals risk, weakening, or uncertainty. Used as the second
# co-occurring group in sentence-level LUI scoring.
# ---------------------------------------------------------------------------
_UNCERTAINTY_TONE_EN = frozenset({
    "uncertain", "uncertainty", "uncertainties",
    "risk", "risks", "risky",
    "downside", "downside risk", "downside risks",
    "weak", "weaken", "weakened", "weakening",
    "soft", "soften", "softened", "softening",
    "decline", "declining", "declined", "declines",
    "deteriorate", "deteriorating", "deteriorated", "deterioration",
    "slow", "slowdown", "slowing", "slowed",
    "fragile", "fragility",
    "concern", "concerns", "concerned",
    "worry", "worries",
    "volatile", "volatility",
    "headwinds",
    "fall", "fell", "falling",
    "drop", "drops", "dropped",
    "loss", "losses",
    "tightening",
    "subdued",
})

_UNCERTAINTY_TONE_ES = frozenset({
    "incierto", "incertidumbre", "incertidumbres",
    "riesgo", "riesgos",
    "riesgos a la baja", "sesgo a la baja",
    "débil", "debilidad", "debilitamiento",
    "blando", "moderación",
    "caída", "caer", "cayendo",
    "deterioro", "deteriorarse",
    "desaceleración", "ralentización",
    "frágil", "fragilidad",
    "preocupación", "preocupaciones", "preocupante",
    "volátil", "volatilidad",
    "vientos en contra",
    "descenso", "bajada",
    "pérdida", "pérdidas",
    "endurecimiento", "tensionamiento",
    "reducido",
})

_UNCERTAINTY_TONE_PT = frozenset({
    "incerto", "incerteza", "incertezas",
    "risco", "riscos",
    "riscos negativos", "viés de baixa",
    "fraco", "fraqueza", "enfraquecimento",
    "moderação",
    "queda", "cair", "caindo",
    "deterioração", "deteriorando",
    "desaceleração",
    "frágil", "fragilidade",
    "preocupação", "preocupações", "preocupante",
    "volátil", "volatilidade",
    "ventos contrários",
    "baixa",
    "perda", "perdas",
    "aperto",
    "subdimensionado",
})

_UNCERTAINTY_TONE_DE = frozenset({
    "unsicher", "unsicherheit", "unsicherheiten",
    "risiko", "risiken",
    "abwärtsrisiko", "abwärtsrisiken",
    "schwach", "schwäche", "abschwächung",
    "weich",
    "rückgang", "rückläufig", "fallen",
    "verschlechterung", "verschlechtern",
    "verlangsamung",
    "fragil", "fragilität",
    "sorge", "sorgen", "besorgnis", "besorgniserregend",
    "volatil", "volatilität",
    "gegenwind",
    "sinken", "sinkend",
    "verlust", "verluste",
    "straffung",
    "gedämpft",
})

_UNCERTAINTY_TONE_FR = frozenset({
    "incertain", "incertitude", "incertitudes",
    "risque", "risques",
    "risques baissiers", "risques à la baisse",
    "faible", "faiblesse", "affaiblissement",
    "ralentissement",
    "baisse", "baisser",
    "détérioration", "se détériorer",
    "fragile", "fragilité",
    "préoccupation", "préoccupations", "préoccupant",
    "volatile", "volatilité",
    "vents contraires",
    "chute", "chuter",
    "perte", "pertes",
    "resserrement",
    "modéré",
})

_UNCERTAINTY_TONE_IT = frozenset({
    "incerto", "incertezza", "incertezze",
    "rischio", "rischi",
    "rischi al ribasso", "rischi negativi",
    "debole", "debolezza", "indebolimento",
    "moderazione",
    "calo", "calare",
    "deterioramento", "peggioramento",
    "rallentamento",
    "fragile", "fragilità",
    "preoccupazione", "preoccupazioni", "preoccupante",
    "volatile", "volatilità",
    "venti contrari",
    "caduta",
    "perdita", "perdite",
    "irrigidimento", "stretta",
    "moderato",
})

_UNCERTAINTY_TONE_JA = frozenset({
    "不確実", "不確実性",
    "リスク", "下振れリスク",
    "弱い", "弱含み", "軟調",
    "減少", "低下",
    "悪化",
    "鈍化", "減速",
    "脆弱",
    "懸念", "憂慮",
    "ボラティリティ",
    "下落",
    "損失",
    "引き締め",
    "低調",
})

_UNCERTAINTY_TONE_ZH = frozenset({
    "不确定", "不确定性",
    "风险", "下行风险",
    "疲弱", "疲软", "疲态",
    "减弱", "走弱",
    "恶化",
    "放缓", "减速",
    "脆弱",
    "担忧", "忧虑",
    "波动", "波动性",
    "下跌",
    "损失",
    "收紧",
    "低迷",
})


# ---------------------------------------------------------------------------
# LUI curated phrases (English) — six conceptual groups, ~150 terms.
# Terms are lowercase phrases; word-boundary regex matched at score time.
# ---------------------------------------------------------------------------
_LUI_PHRASES_EN = frozenset({
    # Group 1: Layoffs
    "layoff", "layoffs", "lay off", "lay offs", "laid off", "laying off",
    "redundancy", "redundancies", "made redundant",
    "downsizing", "downsize", "downsized", "downsizes",
    "workforce reduction", "headcount reduction", "headcount cut",
    "job cuts", "job losses", "job loss",
    "mass layoff", "mass layoffs",
    "reduction in force",
    "dismissal", "dismissed", "dismissals",
    "termination", "terminations",
    "severance package", "severance pay",
    "restructuring", "restructure",
    "pink slip", "pink slips",
    "staff cuts", "staff reductions",
    # Group 2: Hiring freeze
    "hiring freeze", "hiring-freeze", "hiring freezes",
    "hiring pause", "hiring pauses",
    "recruitment freeze", "recruitment pause",
    "headcount freeze",
    "suspended hiring", "paused hiring",
    "hiring slowdown", "slowing hiring", "slowdown in hiring",
    "no new hires", "freeze on new hires",
    "attrition only",
    "selective hiring",
    "hiring halt", "halt hiring",
    # Group 3: Wage compression
    "wage compression", "wage-compression",
    "wage stagnation", "stagnant wages",
    "real wage decline", "declining real wages",
    "wage softening", "softening wages",
    "depressed wages", "suppressed wages",
    "wage moderation",
    "compressed pay",
    "soft wage growth", "weak wage growth",
    "weakening wage pressure",
    "wage decline", "declining wages",
    "pay cuts", "pay cut",
    "wage freeze", "frozen wages",
    # Group 4: Labor shortage
    "labor shortage", "labor-shortage", "labour shortage",
    "skill shortage", "skills shortage", "skills gap",
    "talent shortage", "war for talent",
    "tight labor market", "tight labour market",
    "labor scarcity",
    "hiring difficulty", "difficulty hiring",
    "hard to fill", "hard-to-fill",
    "hiring bottleneck",
    "worker shortage", "workforce shortage",
    "staffing shortage",
    "labor crunch", "labour crunch",
    # Group 5: Participation drop
    "participation rate", "labor force participation",
    "discouraged workers", "discouraged worker",
    "dropout from the labor force", "dropping out of the labor force",
    "decline in participation", "declining participation",
    "sidelined workers",
    "withdrawing from the workforce",
    "exit from the labor force", "labor force exit",
    "nonparticipation",
    "inactive workers",
    "prime-age participation",
    # Group 6: Unemployment risk
    "unemployment", "joblessness",
    "jobless claims", "initial claims", "continuing claims",
    "unemployment risk", "employment risk",
    "rising unemployment", "increased unemployment",
    "weakening employment", "weakening labor market",
    "softening labor market",
    "deteriorating employment", "employment deterioration",
    "employment uncertainty",
    "labor market deterioration",
    "labor market weakness",
    "employment outlook",
    "jobless rate",
})


# ---------------------------------------------------------------------------
# Spanish (es)
# ---------------------------------------------------------------------------
_EPU_ES = {
    "economy":     frozenset({"económica", "económico", "economía"}),
    "policy":      frozenset({"política", "políticas", "regulación",
                              "regulatoria", "legislación", "déficit",
                              "arancel", "aranceles", "banco central"}),
    "uncertainty": frozenset({"incierto", "incierta", "incertidumbre"}),
}
_MPU_ES = frozenset({
    "monetaria", "política", "políticas",
    "banco central", "banco de méxico", "tipo de interés",
    "incierto", "incierta", "incertidumbre", "ambigüedad",
})
_GPR_ES = frozenset({
    "guerra", "militar",
    "terrorismo", "terrorista",
    "geopolítico", "geopolítica",
    "sanciones", "sanción",
    "invasión", "invadir",
    "nuclear", "misil",
    "conflicto", "tensiones",
})
_TONE_ES = {
    "hawkish": frozenset({
        "halcón", "halcones", "endurecer", "endurecimiento",
        "subir", "subió", "aumentó", "aumento",
        "restrictivo", "restrictiva",
        "inflacionario", "inflacionaria",
    }),
    "dovish": frozenset({
        "paloma", "palomas", "relajar", "relajamiento",
        "recortar", "recortó", "bajó", "reducir",
        "acomodaticio", "acomodaticia",
        "estímulo", "apoyo",
    }),
}
_WUI_ES = frozenset({
    "incierto", "incierta", "incertidumbre",
    "ambigüedad", "ambiguo", "ambigua",
    "imprevisible", "impredecible",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "expectativas de inflación", "expectativas ancladas", "expectativas desancladas",
    "desviación del objetivo",
    "efectos de segunda ronda",
    "espiral salarios-precios",
    "riesgos a la baja", "riesgos al alza",
    "riesgos asimétricos",
    "recuperación frágil",
    "perspectivas inciertas",
    "vientos en contra",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (Spanish) — six conceptual groups, ~115 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_ES = frozenset({
    # Group 1: Layoffs (despidos)
    "despido", "despidos", "despedido", "despedidos",
    "indemnización", "indemnizaciones",
    "reducción de personal", "reducción de plantilla",
    "recortes de empleo", "pérdida de empleo", "pérdidas de empleo",
    "despido masivo", "despidos masivos",
    "ajuste de plantilla", "ajuste de personal",
    "rescisión", "rescisión laboral",
    "finiquito",
    "reestructuración", "reestructurar",
    "cierre de empresa", "cierre",
    "expediente de regulación de empleo", "ere",
    "expediente de regulación temporal de empleo", "erte",
    "amortización de puestos", "amortización del puesto",
    "prejubilación", "prejubilaciones",
    "retiro voluntario",
    # Group 2: Hiring freeze
    "congelación de contrataciones", "congelamiento de contrataciones",
    "pausa en contrataciones", "pausa en las contrataciones",
    "freno a la contratación", "freno en la contratación",
    "ralentización de las contrataciones",
    "no contratar", "sin nuevas contrataciones",
    "selectividad en la contratación",
    "cese de contrataciones",
    # Group 3: Wage compression
    "compresión salarial", "compresión de salarios",
    "estancamiento salarial", "salarios estancados",
    "caída salarial real", "caída de salarios reales",
    "moderación salarial",
    "salarios deprimidos", "salarios reprimidos",
    "crecimiento salarial débil",
    "presión salarial débil", "debilitamiento de la presión salarial",
    "recortes salariales", "recorte salarial",
    "congelación salarial", "salarios congelados",
    "rebajas salariales",
    # Group 4: Labor shortage
    "escasez de mano de obra", "escasez laboral",
    "escasez de habilidades", "escasez de capacidades",
    "brecha de habilidades",
    "escasez de talento", "guerra por el talento",
    "mercado laboral ajustado", "mercado laboral tenso",
    "dificultad de contratación", "dificultad para contratar",
    "puestos difíciles de cubrir",
    "cuello de botella en contratación",
    "escasez de trabajadores",
    "vacantes sin cubrir", "vacantes difíciles de cubrir",
    # Group 5: Participation drop
    "tasa de participación", "tasa de participación laboral",
    "participación de la fuerza laboral",
    "trabajadores desalentados",
    "abandono del mercado laboral", "abandono de la fuerza laboral",
    "caída en la participación", "descenso en la participación",
    "trabajadores marginados",
    "salida de la fuerza laboral",
    "no participación", "inactivos",
    "trabajadores inactivos",
    "subempleo",
    # Group 6: Unemployment risk
    "desempleo", "paro", "desocupación",
    "solicitudes de desempleo", "solicitudes iniciales",
    "riesgo de desempleo",
    "aumento del desempleo", "incremento del desempleo",
    "debilitamiento del empleo", "empleo en deterioro",
    "deterioro del mercado laboral",
    "debilidad del mercado laboral",
    "incertidumbre laboral", "incertidumbre del empleo",
    "perspectivas de empleo",
    "tasa de paro", "tasa de desempleo",
    "subutilización del trabajo",
    "contracción del empleo",
    "caída del empleo",
})


# ---------------------------------------------------------------------------
# Portuguese (pt)
# ---------------------------------------------------------------------------
_EPU_PT = {
    "economy":     frozenset({"econômica", "econômico", "economia",
                              "económica", "económico"}),
    "policy":      frozenset({"política", "políticas", "regulação",
                              "regulamentação", "legislação", "défice",
                              "tarifa", "tarifas", "banco central"}),
    "uncertainty": frozenset({"incerto", "incerta", "incerteza"}),
}
_MPU_PT = frozenset({
    "monetária", "política", "políticas",
    "banco central", "banco do brasil", "taxa de juros",
    "incerto", "incerta", "incerteza",
})
_GPR_PT = frozenset({
    "guerra", "militar",
    "terrorismo", "terrorista",
    "geopolítico", "geopolítica",
    "sanções", "sanção",
    "invasão", "invadir",
    "nuclear", "míssil",
    "conflito", "tensões",
})
_TONE_PT = {
    "hawkish": frozenset({
        "falcão", "falcões", "apertar", "aperto",
        "subir", "subiu", "aumentar", "aumento",
        "restritiva", "restritivo",
    }),
    "dovish": frozenset({
        "pomba", "pombas", "relaxar", "relaxamento",
        "cortar", "cortou", "reduzir", "redução",
        "acomodatícia", "acomodatício",
        "estímulo", "apoio",
    }),
}
_WUI_PT = frozenset({
    "incerto", "incerta", "incerteza",
    "ambiguidade", "ambíguo", "ambígua",
    "imprevisível",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "expectativas de inflação", "expectativas ancoradas", "expectativas desancoradas",
    "desvio da meta",
    "efeitos de segunda rodada",
    "espiral salários-preços",
    "riscos negativos", "riscos positivos",
    "riscos assimétricos",
    "recuperação frágil",
    "perspectivas incertas",
    "ventos contrários",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (Portuguese) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_PT = frozenset({
    # Group 1: Layoffs (demissões)
    "demissão", "demissões", "demitido", "demitidos",
    "rescisão", "rescisões", "rescisão contratual",
    "redução de pessoal", "redução de quadro",
    "corte de empregos", "cortes de empregos",
    "perda de emprego", "perdas de emprego",
    "demissão em massa", "demissões em massa",
    "ajuste de quadro",
    "indenização", "indenizações", "indenização rescisória",
    "reestruturação", "reestruturar",
    "fechamento da empresa", "fechamento",
    "desligamento", "desligamentos",
    "corte de pessoal", "cortes de pessoal",
    "enxugamento",
    "programa de demissão voluntária", "pdv",
    "demissão consensual",
    "aposentadoria antecipada",
    # Group 2: Hiring freeze
    "congelamento de contratações", "congelamento das contratações",
    "pausa nas contratações", "pausa em contratações",
    "freio nas contratações",
    "desaceleração das contratações",
    "sem novas contratações", "não contratar",
    "contratação seletiva",
    "suspensão das contratações",
    # Group 3: Wage compression
    "compressão salarial",
    "estagnação salarial", "salários estagnados",
    "queda real dos salários",
    "moderação salarial",
    "salários deprimidos",
    "crescimento salarial fraco",
    "pressão salarial fraca",
    "cortes salariais", "corte salarial",
    "congelamento salarial", "salários congelados",
    "redução salarial", "redução dos salários",
    "estagnação dos salários",
    # Group 4: Labor shortage
    "escassez de mão de obra", "escassez laboral",
    "escassez de habilidades",
    "escassez de talento", "guerra por talento",
    "mercado de trabalho apertado", "mercado de trabalho aquecido",
    "dificuldade de contratação", "dificuldade em contratar",
    "vagas difíceis de preencher",
    "gargalo de contratação",
    "escassez de trabalhadores",
    "vagas ociosas", "vagas em aberto",
    "déficit de mão de obra",
    # Group 5: Participation drop
    "taxa de participação", "taxa de participação no mercado de trabalho",
    "participação na força de trabalho",
    "trabalhadores desencorajados",
    "saída da força de trabalho", "afastamento da força de trabalho",
    "queda na participação", "diminuição da participação",
    "inatividade", "inativos",
    "trabalhadores inativos",
    "subutilização da força de trabalho",
    "subemprego",
    # Group 6: Unemployment risk
    "desemprego",
    "pedidos de seguro-desemprego", "solicitações de seguro",
    "risco de desemprego",
    "alta do desemprego", "aumento do desemprego",
    "enfraquecimento do emprego",
    "deterioração do mercado de trabalho",
    "fraqueza do mercado de trabalho",
    "incerteza no emprego", "incerteza do mercado de trabalho",
    "perspectivas de emprego",
    "taxa de desemprego",
    "desemprego juvenil",
    "contração do emprego",
    "queda do emprego",
})


# ---------------------------------------------------------------------------
# German (de)
# ---------------------------------------------------------------------------
_EPU_DE = {
    "economy":     frozenset({"wirtschaftlich", "wirtschaft"}),
    "policy":      frozenset({"politik", "regulierung", "gesetzgebung",
                              "defizit", "zoll", "zentralbank",
                              "europäische zentralbank", "bundestag"}),
    "uncertainty": frozenset({"unsicher", "unsicherheit"}),
}
_MPU_DE = frozenset({
    "geldpolitik", "geldpolitisch",
    "zentralbank", "europäische zentralbank", "bundesbank",
    "leitzins", "zinssatz",
    "unsicher", "unsicherheit",
})
_GPR_DE = frozenset({
    "krieg", "militär",
    "terror", "terrorismus", "terroristisch",
    "geopolitisch", "geopolitik",
    "sanktionen", "sanktion",
    "invasion",
    "nuklear", "rakete",
    "konflikt", "spannungen",
})
_TONE_DE = {
    "hawkish": frozenset({
        "falke", "falken", "straffen", "straffung",
        "anheben", "erhöhen", "erhöhung",
        "restriktiv", "inflationär",
    }),
    "dovish": frozenset({
        "taube", "tauben", "lockern", "lockerung",
        "senken", "senkung",
        "akkommodativ", "stimulus", "unterstützung",
    }),
}
_WUI_DE = frozenset({
    "unsicher", "unsicherheit",
    "uneindeutig", "ambiguität",
    "unvorhersehbar",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "inflationserwartungen", "verankerte erwartungen", "entankerte erwartungen",
    "abweichung vom ziel",
    "zweitrundeneffekte",
    "lohn-preis-spirale",
    "abwärtsrisiken", "aufwärtsrisiken",
    "asymmetrische risiken",
    "fragile erholung",
    "unsicherer ausblick",
    "gegenwind",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (German) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_DE = frozenset({
    # Group 1: Layoffs (Entlassungen)
    "entlassung", "entlassungen", "entlassen",
    "kündigung", "kündigungen",
    "personalabbau", "stellenabbau",
    "arbeitsplatzverluste", "arbeitsplatzverlust",
    "massenentlassung", "massenentlassungen",
    "abfindung", "abfindungen",
    "umstrukturierung", "restrukturierung",
    "betriebsschließung", "werkschließung",
    "freisetzung", "freisetzungen",
    "personalfreisetzung",
    "betriebsbedingte kündigung", "betriebsbedingte kündigungen",
    "abbau von stellen", "stellenstreichung", "stellenstreichungen",
    "vorruhestand", "altersteilzeit",
    # Group 2: Hiring freeze
    "einstellungsstopp", "einstellungssperre",
    "neueinstellungsstopp",
    "verlangsamung der einstellungen",
    "keine neueinstellungen",
    "selektive einstellung",
    "einstellungspause",
    "verhaltene einstellungen",
    # Group 3: Wage compression
    "lohnstagnation", "lohnstagnierung",
    "reallohnverlust", "reallohnrückgang",
    "lohnzurückhaltung", "lohnmoderation",
    "schwaches lohnwachstum",
    "schwacher lohndruck", "nachlassender lohndruck",
    "lohnkürzung", "lohnkürzungen",
    "lohneinfrierung", "eingefrorene löhne",
    "kurzarbeit", "kurzarbeitergeld",
    "tarifabschluss schwach", "lohnzurückhaltung der gewerkschaften",
    "reallohnverlust der arbeitnehmer",
    "lohnpause",
    # Group 4: Labor shortage
    "arbeitskräftemangel", "fachkräftemangel",
    "qualifikationsmangel",
    "talentmangel",
    "angespannter arbeitsmarkt", "enger arbeitsmarkt",
    "einstellungsschwierigkeiten",
    "schwer zu besetzen",
    "personalknappheit",
    "qualifizierter fachkräftemangel", "ingenieurmangel", "pflegekräftemangel",
    "vakanzquote hoch", "schwer zu besetzende stellen",
    # Group 5: Participation drop
    "erwerbsquote", "erwerbsbeteiligung",
    "entmutigte arbeitnehmer",
    "rückzug aus dem arbeitsmarkt", "ausscheiden aus dem erwerbsleben",
    "rückgang der erwerbsbeteiligung",
    "nichterwerbstätigkeit",
    "stille reserve",
    "stille reserve am arbeitsmarkt",
    "unterbeschäftigung",
    # Group 6: Unemployment risk
    "arbeitslosigkeit", "erwerbslosigkeit",
    "arbeitslosenanträge",
    "arbeitslosigkeitsrisiko",
    "anstieg der arbeitslosigkeit", "zunehmende arbeitslosigkeit",
    "abschwächung des arbeitsmarktes",
    "verschlechterung des arbeitsmarktes",
    "schwäche des arbeitsmarktes",
    "beschäftigungsunsicherheit", "arbeitsmarktunsicherheit",
    "beschäftigungsausblick",
    "arbeitslosenquote",
    "arbeitslosenzahl", "anzahl der arbeitslosen",
    "anstieg arbeitsloser", "arbeitsmarktdaten verschlechtern",
    "jobverlust", "jobverluste",
    "beschäftigungsrückgang", "entlassungswelle", "stellenkürzung",
})


# ---------------------------------------------------------------------------
# French (fr)
# ---------------------------------------------------------------------------
_EPU_FR = {
    "economy":     frozenset({"économique", "économie"}),
    "policy":      frozenset({"politique", "politiques", "réglementation",
                              "législation", "déficit", "tarif", "douane",
                              "banque centrale", "banque de france"}),
    "uncertainty": frozenset({"incertain", "incertaine", "incertitude"}),
}
_MPU_FR = frozenset({
    "monétaire", "politique", "politiques",
    "banque centrale", "banque de france",
    "taux directeur", "taux d'intérêt",
    "incertain", "incertaine", "incertitude",
})
_GPR_FR = frozenset({
    "guerre", "militaire",
    "terrorisme", "terroriste",
    "géopolitique",
    "sanctions", "sanction",
    "invasion",
    "nucléaire", "missile",
    "conflit", "tensions",
})
_TONE_FR = {
    "hawkish": frozenset({
        "faucon", "faucons", "durcir", "durcissement",
        "relever", "relèvement", "hausse",
        "restrictif", "restrictive",
    }),
    "dovish": frozenset({
        "colombe", "colombes", "assouplir", "assouplissement",
        "baisse", "baisser", "abaisser",
        "accommodant", "accommodante",
        "soutien",
    }),
}
_WUI_FR = frozenset({
    "incertain", "incertaine", "incertitude",
    "ambiguïté", "ambigu", "ambiguë",
    "imprévisible",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "anticipations d'inflation", "anticipations ancrées", "anticipations désancrées",
    "écart à la cible",
    "effets de second tour",
    "spirale salaires-prix",
    "risques baissiers", "risques haussiers",
    "risques asymétriques",
    "reprise fragile",
    "perspectives incertaines",
    "vents contraires",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (French) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_FR = frozenset({
    # Group 1: Layoffs (licenciements)
    "licenciement", "licenciements", "licencié", "licenciés",
    "rupture de contrat", "ruptures de contrat",
    "indemnité de licenciement", "indemnités",
    "réduction d'effectifs", "réduction du personnel",
    "suppression de postes", "suppressions de postes",
    "perte d'emploi", "pertes d'emploi",
    "plan social", "plans sociaux",
    "licenciement économique", "licenciements économiques",
    "restructuration", "restructurations",
    "fermeture d'entreprise", "fermeture de l'usine",
    "plan de sauvegarde de l'emploi", "pse",
    "rupture conventionnelle", "ruptures conventionnelles",
    "départ volontaire", "départs volontaires",
    "préretraite", "préretraites",
    # Group 2: Hiring freeze
    "gel des embauches", "gel de l'embauche",
    "pause des embauches", "pause dans les embauches",
    "ralentissement des embauches",
    "pas de nouvelles embauches",
    "embauche sélective",
    "chômage partiel", "activité partielle",
    # Group 3: Wage compression
    "compression salariale",
    "stagnation salariale", "salaires stagnants",
    "baisse réelle des salaires",
    "modération salariale",
    "salaires déprimés",
    "croissance salariale faible",
    "pression salariale faible",
    "baisse des salaires", "réduction des salaires",
    "gel des salaires", "salaires gelés",
    "stagnation des salaires", "baisse de pouvoir d'achat",
    "modération des salaires",
    # Group 4: Labor shortage
    "pénurie de main-d'œuvre", "pénurie de main d'oeuvre",
    "pénurie de compétences",
    "pénurie de talents", "guerre des talents",
    "marché du travail tendu",
    "difficultés de recrutement", "difficulté à recruter",
    "postes difficiles à pourvoir",
    "manque de personnel",
    "tension de recrutement", "tensions de recrutement",
    "métiers en tension",
    "main-d'œuvre rare", "manque de main-d'œuvre",
    # Group 5: Participation drop
    "taux d'activité", "taux de participation",
    "participation au marché du travail",
    "travailleurs découragés",
    "sortie du marché du travail", "abandon du marché du travail",
    "baisse de la participation", "recul de la participation",
    "inactivité",
    "sous-emploi",
    "halo du chômage",
    "emploi précaire", "précarité de l'emploi",
    # Group 6: Unemployment risk
    "chômage",
    "demandes d'allocations chômage", "inscriptions au chômage",
    "risque de chômage",
    "hausse du chômage", "augmentation du chômage",
    "affaiblissement de l'emploi",
    "détérioration du marché du travail",
    "faiblesse du marché du travail",
    "incertitude de l'emploi", "incertitude sur l'emploi",
    "perspectives d'emploi",
    "taux de chômage",
    "demandeurs d'emploi", "inscrits à pôle emploi",
    "détérioration de l'emploi", "fragilité du marché du travail",
    "chômage des jeunes", "chômage de longue durée",
})


# ---------------------------------------------------------------------------
# Italian (it)
# ---------------------------------------------------------------------------
_EPU_IT = {
    "economy":     frozenset({"economica", "economico", "economia"}),
    "policy":      frozenset({"politica", "politiche", "regolamentazione",
                              "legislazione", "deficit", "tariffa",
                              "banca centrale"}),
    "uncertainty": frozenset({"incerto", "incerta", "incertezza"}),
}
_MPU_IT = frozenset({
    "monetaria", "politica", "politiche",
    "banca centrale", "bce",
    "tasso", "tassi",
    "incerto", "incerta", "incertezza",
})
_GPR_IT = frozenset({
    "guerra", "militare",
    "terrorismo", "terrorista",
    "geopolitico", "geopolitica",
    "sanzioni", "sanzione",
    "invasione",
    "nucleare", "missile",
    "conflitto", "tensioni",
})
_TONE_IT = {
    "hawkish": frozenset({
        "falco", "falchi", "stringere", "stretta",
        "alzare", "rialzo", "aumento",
        "restrittiva", "restrittivo",
    }),
    "dovish": frozenset({
        "colomba", "colombe", "allentare", "allentamento",
        "ridurre", "riduzione", "abbassare",
        "accomodante", "stimolo", "sostegno",
    }),
}
_WUI_IT = frozenset({
    "incerto", "incerta", "incertezza",
    "ambiguità", "ambiguo", "ambigua",
    "imprevedibile",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "aspettative di inflazione", "aspettative ancorate", "aspettative disancorate",
    "scostamento dall'obiettivo",
    "effetti di secondo round",
    "spirale salari-prezzi",
    "rischi al ribasso", "rischi al rialzo",
    "rischi asimmetrici",
    "ripresa fragile",
    "prospettive incerte",
    "venti contrari",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (Italian) — six conceptual groups, ~105 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_IT = frozenset({
    # Group 1: Layoffs (licenziamenti)
    "licenziamento", "licenziamenti", "licenziato", "licenziati",
    "risoluzione del rapporto",
    "tfr", "trattamento di fine rapporto",
    "riduzione del personale", "riduzione di organico",
    "tagli al personale", "tagli occupazionali",
    "perdita del lavoro", "perdite di lavoro",
    "licenziamenti collettivi",
    "esuberi", "esubero",
    "ristrutturazione", "ristrutturazioni",
    "chiusura aziendale", "chiusura dello stabilimento",
    "cassa integrazione", "cassa integrazione guadagni", "cig", "cigs",
    "mobilità", "lista di mobilità",
    "ammortizzatori sociali",
    "esodo incentivato", "incentivo all'esodo",
    "prepensionamento", "prepensionamenti",
    # Group 2: Hiring freeze
    "blocco delle assunzioni", "blocco assunzioni",
    "pausa nelle assunzioni",
    "rallentamento delle assunzioni",
    "nessuna nuova assunzione",
    "assunzioni selettive",
    # Group 3: Wage compression
    "compressione salariale",
    "stagnazione salariale", "salari stagnanti",
    "calo reale dei salari",
    "moderazione salariale",
    "salari depressi",
    "crescita salariale debole",
    "pressione salariale debole",
    "tagli salariali", "taglio salariale",
    "congelamento salariale", "salari congelati",
    "blocco salariale",
    "moderazione salariale",
    # Group 4: Labor shortage
    "carenza di manodopera",
    "carenza di competenze",
    "carenza di talenti",
    "mercato del lavoro teso",
    "difficoltà di assunzione", "difficoltà ad assumere",
    "posti difficili da riempire",
    "mancanza di personale",
    "tensione sul mercato del lavoro",
    "carenza di manodopera qualificata",
    "vacancy difficili da coprire",
    # Group 5: Participation drop
    "tasso di partecipazione", "tasso di attività",
    "partecipazione al mercato del lavoro",
    "lavoratori scoraggiati",
    "uscita dal mercato del lavoro", "abbandono del mercato del lavoro",
    "calo della partecipazione", "diminuzione della partecipazione",
    "inattività",
    "lavoro precario", "precarietà",
    "sottoccupazione",
    # Group 6: Unemployment risk
    "disoccupazione",
    "richieste di disoccupazione", "domande di disoccupazione",
    "rischio di disoccupazione",
    "aumento della disoccupazione", "incremento della disoccupazione",
    "indebolimento dell'occupazione",
    "deterioramento del mercato del lavoro",
    "debolezza del mercato del lavoro",
    "incertezza occupazionale",
    "prospettive occupazionali",
    "tasso di disoccupazione",
    "disoccupati di lunga durata",
    "domande di disoccupazione in aumento",
    "deterioramento del lavoro", "fragilità occupazionale",
    "neet", "disoccupazione giovanile",
    "perdita di lavoro", "perdita occupazionale",
    "crisi occupazionale", "contrazione occupazionale",
    "blocco del turnover",
    "congelamento delle assunzioni",
    "flessione occupazionale", "riduzione dell'occupazione",
    "mercato del lavoro debole",
})


# ---------------------------------------------------------------------------
# Japanese (ja) — substring matching, no whitespace tokenization
# ---------------------------------------------------------------------------
_EPU_JA = {
    "economy":     frozenset({"経済"}),
    "policy":      frozenset({"政策", "規制"}),
    "uncertainty": frozenset({"不確実", "不確実性"}),
}
_MPU_JA = frozenset({"金融政策", "中央銀行", "日本銀行", "金利", "不確実性"})
_GPR_JA = frozenset({"戦争", "軍事", "テロ", "地政学", "制裁", "侵攻", "核", "ミサイル", "紛争"})
_WUI_JA = frozenset({
    "不確実", "不確実性", "曖昧",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "インフレ期待",
    "アンカーされた期待", "アンカー外れ",
    "目標からの乖離",
    "第二次効果", "二次的効果",
    "賃金物価スパイラル",
    "下振れリスク", "上振れリスク",
    "非対称的リスク",
    "脆弱な回復",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (Japanese) — six conceptual groups, ~65 terms.
# Japanese economic vocabulary for labor uncertainty has tighter
# concept-to-term mapping than Latin scripts.
# ---------------------------------------------------------------------------
_LUI_PHRASES_JA = frozenset({
    # Group 1: Layoffs (解雇 / リストラ)
    "解雇", "解雇者", "整理解雇",
    "リストラ", "リストラクチャリング", "リストラされ",
    "人員削減", "人員整理", "人員カット",
    "首切り",
    "退職勧奨", "退職金",
    "希望退職",
    "雇い止め",
    "事業所閉鎖", "工場閉鎖",
    # Group 2: Hiring freeze
    "採用凍結", "採用停止", "採用見送り",
    "雇用凍結",
    "新規採用停止",
    "採用抑制",
    "新卒採用見送り",
    "中途採用抑制",
    # Group 3: Wage compression
    "賃金停滞", "賃金抑制",
    "実質賃金低下",
    "賃金カット", "賃金引き下げ",
    "賃上げ抑制",
    "賃金凍結",
    "賞与カット", "ボーナスカット", "賞与減額",
    "ベースアップ抑制",
    # Group 4: Labor shortage
    "労働力不足", "人手不足", "人材不足",
    "技能不足", "スキル不足",
    "売り手市場",
    "採用難",
    "深刻な人手不足",
    "応募者不足",
    # Group 5: Participation drop
    "労働参加率",
    "求職活動の停止", "労働市場からの退出",
    "非労働力化",
    # Group 6: Unemployment risk
    "失業", "失職", "離職",
    "失業給付申請", "失業保険申請",
    "失業リスク",
    "失業率上昇", "失業の増加",
    "雇用悪化", "雇用情勢の悪化",
    "雇用不安",
    "労働市場の弱さ",
    "失業率",
    "完全失業率",
    "求人倍率低下",
    "雇用情勢の厳しさ",
})
_TONE_JA = {
    "hawkish": frozenset({"引き締め", "利上げ", "タカ派", "緊縮"}),
    "dovish":  frozenset({"緩和", "利下げ", "ハト派", "刺激"}),
}


# ---------------------------------------------------------------------------
# Chinese (zh) — substring matching, simplified-Chinese terms
# ---------------------------------------------------------------------------
_EPU_ZH = {
    "economy":     frozenset({"经济", "经济的"}),
    "policy":      frozenset({"政策", "监管", "法规"}),
    "uncertainty": frozenset({"不确定", "不确定性"}),
}
_MPU_ZH = frozenset({"货币政策", "中央银行", "人民银行", "利率", "不确定性"})
_GPR_ZH = frozenset({"战争", "军事", "恐怖", "地缘政治", "制裁", "入侵", "核", "导弹", "冲突"})
_WUI_ZH = frozenset({
    "不确定", "不确定性", "模糊",
    # Slice 6a additions — Hubert-inspired vocabulary.
    "通胀预期",
    "锚定预期", "脱锚预期",
    "偏离目标",
    "二轮效应", "次轮效应",
    "工资价格螺旋",
    "下行风险", "上行风险",
    "不对称风险",
    "复苏脆弱",
})
# ---------------------------------------------------------------------------
# LUI curated phrases (Simplified Chinese) — six conceptual groups,
# ~65 terms.
# ---------------------------------------------------------------------------
_LUI_PHRASES_ZH = frozenset({
    # Group 1: Layoffs (裁员)
    "裁员", "裁员潮",
    "解雇", "解聘",
    "辞退", "辞退员工",
    "人员精简", "精简人员",
    "下岗", "下岗职工",
    "经济补偿", "经济补偿金",
    "重组", "重组裁员",
    "工厂关闭", "厂房关闭", "停产",
    # Group 2: Hiring freeze
    "招聘冻结", "招聘暂停",
    "停止招聘",
    "招聘放缓",
    "暂停招新",
    "选择性招聘",
    "稳就业", "保就业",
    # Group 3: Wage compression
    "工资停滞", "薪资停滞",
    "实际工资下降",
    "工资降低", "降薪",
    "工资增长缓慢", "工资增长疲软",
    "工资压力减弱",
    "工资冻结",
    "工资增长放缓", "薪资增速放缓",
    # Group 4: Labor shortage
    "劳动力短缺", "用工短缺",
    "技能短缺", "技能不足",
    "人才短缺", "人才争夺战",
    "就业市场紧张",
    "招聘困难",
    "招工难", "用工难",
    # Group 5: Participation drop
    "劳动参与率",
    "退出劳动力市场", "退出劳动市场",
    "气馁工人", "受挫工人",
    "非劳动人口",
    "灵活就业", "灵活就业人员",
    # Group 6: Unemployment risk
    "失业", "失业潮",
    "失业救济金申请",
    "失业风险",
    "失业率上升", "失业增加",
    "就业疲软", "就业恶化",
    "劳动力市场恶化",
    "就业不确定性",
    "失业率",
    "青年失业率",
    "就业形势严峻",
    "就业压力",
})
_TONE_ZH = {
    "hawkish": frozenset({"紧缩", "加息", "鹰派", "收紧"}),
    "dovish":  frozenset({"宽松", "降息", "鸽派", "刺激"}),
}


# ---------------------------------------------------------------------------
# Technological-domain vocabulary (Slice b) — AI / automation / robotics /
# digitisation terms. Used as the third co-occurring group in LTUI scoring
# (alongside _LABOR_DOMAIN_<lang> and _UNCERTAINTY_TONE_<lang>).
# Seed list compiled from Baker-Bloom-Davis AI-EPU (2023) and
# Felten-Raj-Seamans (2021) vocabularies.
# Reviewer log: puremacro/docs/lexicon_review.md
# ---------------------------------------------------------------------------
_TECH_DOMAIN_EN = frozenset({
    # AI & machine learning
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural networks", "large language model",
    "large language models", "llm", "llms", "generative ai", "genai",
    "chatgpt", "transformer", "transformers", "foundation model",
    "foundation models", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "predictive analytics",
    # Automation & robotics
    "automation", "automate", "automated", "automating",
    "robot", "robots", "robotic", "robotics", "robotization",
    "industrial robot", "industrial robots",
    "process automation", "rpa", "cobots",
    # Digitisation & platforms
    "digitalisation", "digitalization", "digitisation", "digitization",
    "digital transformation", "digital economy",
    "platform economy", "gig economy", "platform workers",
    "algorithmic management", "algorithms",
    # Compute / data infrastructure
    "cloud computing", "data centre", "data centres", "data center", "data centers",
    "computing power", "compute capacity",
    "big data",
    # Adoption / diffusion language
    "tech adoption", "technology adoption", "technological change",
    "technological progress", "innovation", "diffusion",
    "augmented", "augmentation",
    # Disruption framing
    "disruption", "disruptive technology", "disruptive technologies",
    "technological displacement", "technological unemployment",
    # Specific domains affecting labor
    "self-driving", "autonomous vehicle", "autonomous vehicles",
    "3d printing", "additive manufacturing",
    "biotech", "biotechnology",
    "fintech", "regtech",
})

_TECH_DOMAIN_ES = frozenset({
    "ia", "inteligencia artificial",
    "aprendizaje automático", "aprendizaje profundo",
    "redes neuronales", "modelos de lenguaje", "modelo de lenguaje",
    "chatgpt", "ia generativa", "transformadores",
    "automatización", "automatizar", "automatizado",
    "robot", "robots", "robótica", "robotización",
    "robots industriales",
    "digitalización", "transformación digital", "economía digital",
    "economía de plataformas", "trabajadores de plataforma",
    "computación en la nube", "centros de datos", "big data",
    "adopción tecnológica", "cambio tecnológico", "progreso tecnológico",
    "innovación", "difusión tecnológica",
    "disrupción", "disrupción tecnológica",
    "desempleo tecnológico", "desplazamiento tecnológico",
    "vehículos autónomos", "impresión 3d",
    "biotecnología", "fintech",
})

_TECH_DOMAIN_PT = frozenset({
    "ia", "inteligência artificial",
    "aprendizado de máquina", "aprendizagem profunda",
    "redes neurais", "modelos de linguagem", "modelo de linguagem",
    "chatgpt", "ia generativa", "transformadores",
    "automação", "automatizar", "automatizado",
    "robô", "robôs", "robótica", "robotização",
    "robôs industriais",
    "digitalização", "transformação digital", "economia digital",
    "economia de plataformas", "trabalhadores de plataforma",
    "computação em nuvem", "data centers", "big data",
    "adoção tecnológica", "mudança tecnológica", "progresso tecnológico",
    "inovação", "difusão tecnológica",
    "disrupção", "disrupção tecnológica",
    "desemprego tecnológico", "deslocamento tecnológico",
    "veículos autônomos", "impressão 3d",
    "biotecnologia", "fintech",
})

_TECH_DOMAIN_DE = frozenset({
    "ki", "künstliche intelligenz",
    "maschinelles lernen", "tiefes lernen", "deep learning",
    "neuronale netze", "sprachmodelle", "sprachmodell",
    "chatgpt", "generative ki", "transformer",
    "automatisierung", "automatisieren", "automatisiert",
    "roboter", "robotik", "robotisierung",
    "industrieroboter",
    "digitalisierung", "digitale transformation", "digitale wirtschaft",
    "plattformökonomie", "plattformarbeiter",
    "cloud-computing", "rechenzentren", "big data",
    "technologieadoption", "technologischer wandel", "technologischer fortschritt",
    "innovation", "technologische diffusion",
    "disruption", "technologische disruption",
    "technologische arbeitslosigkeit", "technologische verdrängung",
    "autonome fahrzeuge", "3d-druck",
    "biotechnologie", "fintech",
})

_TECH_DOMAIN_FR = frozenset({
    "ia", "intelligence artificielle",
    "apprentissage automatique", "apprentissage profond",
    "réseaux neuronaux", "modèles de langage", "modèle de langage",
    "chatgpt", "ia générative", "transformeurs",
    "automatisation", "automatiser", "automatisé",
    "robot", "robots", "robotique", "robotisation",
    "robots industriels",
    "numérisation", "transformation numérique", "économie numérique",
    "économie des plateformes", "travailleurs des plateformes",
    "informatique en nuage", "centres de données", "big data",
    "adoption technologique", "changement technologique", "progrès technologique",
    "innovation", "diffusion technologique",
    "disruption", "disruption technologique",
    "chômage technologique", "déplacement technologique",
    "véhicules autonomes", "impression 3d",
    "biotechnologie", "fintech",
})

_TECH_DOMAIN_IT = frozenset({
    "ia", "intelligenza artificiale",
    "apprendimento automatico", "apprendimento profondo",
    "reti neurali", "modelli linguistici", "modello linguistico",
    "chatgpt", "ia generativa", "transformer",
    "automazione", "automatizzare", "automatizzato",
    "robot", "robotica", "robotizzazione",
    "robot industriali",
    "digitalizzazione", "trasformazione digitale", "economia digitale",
    "economia delle piattaforme", "lavoratori delle piattaforme",
    "cloud computing", "centri dati", "big data",
    "adozione tecnologica", "cambiamento tecnologico", "progresso tecnologico",
    "innovazione", "diffusione tecnologica",
    "disruption", "disruption tecnologica",
    "disoccupazione tecnologica", "spostamento tecnologico",
    "veicoli autonomi", "stampa 3d",
    "biotecnologia", "fintech",
})

_TECH_DOMAIN_JA = frozenset({
    "ai", "人工知能",
    "機械学習", "深層学習", "ディープラーニング",
    "ニューラルネットワーク", "大規模言語モデル", "言語モデル",
    "チャットgpt", "生成ai",
    "自動化", "自動化する", "オートメーション",
    "ロボット", "ロボティクス", "ロボット化",
    "産業用ロボット",
    "デジタル化", "デジタルトランスフォーメーション", "デジタル経済",
    "プラットフォーム経済", "プラットフォーム労働者",
    "クラウドコンピューティング", "データセンター", "ビッグデータ",
    "技術導入", "技術変化", "技術進歩",
    "イノベーション", "技術普及",
    "破壊的技術", "技術的破壊",
    "技術的失業", "技術的代替",
    "自動運転車", "3dプリンティング",
    "バイオテクノロジー", "フィンテック",
})

_TECH_DOMAIN_ZH = frozenset({
    "ai", "人工智能",
    "机器学习", "深度学习",
    "神经网络", "大语言模型", "语言模型",
    "chatgpt", "生成式人工智能", "生成式ai",
    "自动化", "自动化生产",
    "机器人", "机器人技术", "机器人化",
    "工业机器人",
    "数字化", "数字化转型", "数字经济",
    "平台经济", "平台工人",
    "云计算", "数据中心", "大数据",
    "技术采用", "技术变革", "技术进步",
    "创新", "技术扩散",
    "颠覆性技术", "技术颠覆",
    "技术性失业", "技术替代",
    "自动驾驶汽车", "3d打印",
    "生物技术", "金融科技",
})


# ---------------------------------------------------------------------------
# War / geopolitical-conflict labor domain (Slice e).
# Extends the English-only _GPR_EN to 8 languages and adds conscription /
# mobilisation / refugee / supply-chain-disruption terms specifically
# relevant to labor disruption. Used as the third co-occurring group in
# LWUI scoring (labor × uncertainty × war).
# Reviewer log: puremacro/docs/lexicon_review.md
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Wage domain — pay-, wage-, compensation-, bargaining-related terms used by
# the LWUI-wage variant (labor × uncertainty × wage). Distinct from
# _WAR_DOMAIN below; the two domains are mutually exclusive on purpose so
# that lwui_wage and lwui_war are not collinear.
# ---------------------------------------------------------------------------
_WAGE_DOMAIN_EN = frozenset({
    "wage", "wages", "salary", "salaries", "pay", "compensation",
    "remuneration", "earnings", "earned income", "income",
    "minimum wage", "living wage", "hourly wage", "median wage",
    "wage growth", "wage inflation", "wage pressure", "wage pressures",
    "wage stagnation", "wage decline", "wage cut", "wage cuts",
    "wage freeze", "pay freeze", "pay cut", "pay cuts",
    "raise", "raises", "pay rise", "pay raises", "pay increase",
    "real wage", "real wages", "nominal wage", "nominal wages",
    "unit labor cost", "unit labour cost", "unit labor costs",
    "collective bargaining", "wage bargaining", "wage negotiation",
    "wage settlement", "wage settlements", "wage agreement",
    "union", "unions", "unionised", "unionized", "trade union",
    "labour union", "labor union", "strike action",
    "cost of living", "cost-of-living adjustment", "cola",
    "tipped wage", "overtime pay", "bonus", "bonuses",
    "minimum-wage increase", "wage floor",
})

_WAGE_DOMAIN_ES = frozenset({
    "salario", "salarios", "sueldo", "sueldos", "paga", "pagas",
    "remuneración", "remuneraciones", "compensación", "ingresos",
    "salario mínimo", "salario digno", "salario por hora",
    "crecimiento salarial", "presión salarial", "presiones salariales",
    "aumento salarial", "aumento de sueldo", "alza salarial",
    "recorte salarial", "rebaja salarial", "congelación salarial",
    "salario real", "salario nominal",
    "costo laboral unitario", "coste laboral unitario",
    "negociación colectiva", "negociación salarial", "convenio colectivo",
    "convenio salarial", "acuerdo salarial",
    "sindicato", "sindicatos", "sindicalizado", "sindicalizada",
    "costo de vida", "coste de vida", "indización salarial",
})

_WAGE_DOMAIN_PT = frozenset({
    "salário", "salários", "remuneração", "remunerações",
    "compensação", "rendimento", "rendimentos", "ordenado",
    "salário mínimo", "salário digno",
    "crescimento salarial", "pressão salarial",
    "aumento salarial", "reajuste salarial",
    "corte salarial", "congelamento salarial",
    "salário real", "salário nominal",
    "custo laboral unitário", "custo unitário do trabalho",
    "negociação coletiva", "acordo coletivo", "convenção coletiva",
    "sindicato", "sindicatos", "sindicalizado", "sindicalizada",
    "custo de vida",
})

_WAGE_DOMAIN_DE = frozenset({
    "lohn", "löhne", "gehalt", "gehälter", "vergütung",
    "entlohnung", "entgelt", "verdienst", "einkommen",
    "mindestlohn", "tariflohn", "stundenlohn", "monatslohn",
    "lohnwachstum", "lohnentwicklung", "lohndruck",
    "lohnerhöhung", "gehaltserhöhung", "lohnsteigerung",
    "lohnkürzung", "lohnsenkung", "lohnstopp",
    "reallohn", "reallöhne", "nominallohn",
    "lohnstückkosten", "tarifverhandlung", "tarifverhandlungen",
    "tarifvertrag", "tarifabschluss",
    "gewerkschaft", "gewerkschaften",
    "lebenshaltungskosten",
})

_WAGE_DOMAIN_FR = frozenset({
    "salaire", "salaires", "rémunération", "rémunérations",
    "paie", "paye", "revenu", "revenus", "traitement",
    "salaire minimum", "smic",
    "croissance salariale", "pression salariale",
    "augmentation salariale", "hausse salariale",
    "baisse salariale", "gel des salaires", "gel salarial",
    "salaire réel", "salaire nominal",
    "coût unitaire du travail",
    "négociation collective", "négociation salariale",
    "accord salarial", "convention collective",
    "syndicat", "syndicats", "syndiqué", "syndiqués",
    "coût de la vie",
})

_WAGE_DOMAIN_IT = frozenset({
    "salario", "salari", "stipendio", "stipendi",
    "retribuzione", "retribuzioni", "compenso",
    "salario minimo", "salario orario",
    "crescita salariale", "pressione salariale",
    "aumento salariale", "rivalutazione salariale",
    "taglio salariale", "congelamento salariale",
    "salario reale", "salario nominale",
    "costo del lavoro unitario",
    "contrattazione collettiva", "negoziazione salariale",
    "contratto collettivo", "accordo salariale",
    "sindacato", "sindacati", "iscritti al sindacato",
    "costo della vita",
})

_WAGE_DOMAIN_JA = frozenset({
    "賃金", "給与", "給料", "報酬", "所得", "収入",
    "最低賃金", "時給", "月給", "年収",
    "賃上げ", "ベースアップ", "ベア",
    "賃下げ", "賃金カット", "賃金凍結",
    "実質賃金", "名目賃金",
    "単位労働コスト", "労働コスト",
    "団体交渉", "賃金交渉", "労使交渉", "労使合意",
    "労働組合", "労組",
    "生計費",
})

_WAGE_DOMAIN_ZH = frozenset({
    "工资", "薪资", "薪水", "薪酬", "薪金",
    "报酬", "收入", "所得",
    "最低工资", "时薪", "月薪",
    "工资增长", "工资上涨", "涨薪", "加薪",
    "降薪", "减薪", "薪资冻结",
    "实际工资", "名义工资",
    "单位劳动成本", "劳动力成本",
    "集体谈判", "工资谈判", "劳资谈判", "集体合同",
    "工会",
    "生活成本",
})


_WAR_DOMAIN_EN = frozenset({
    "war", "warfare", "military", "armed conflict",
    "terror", "terrorism", "terrorist", "terrorists",
    "geopolitical", "geopolitics",
    "sanctions", "sanction", "trade embargo", "embargo",
    "invasion", "invade", "invading",
    "nuclear", "missile", "missiles",
    "conflict", "conflicts", "tensions",
    "mobilisation", "mobilization", "mobilise", "mobilize",
    "conscription", "draft", "conscript", "conscripts",
    "refugee", "refugees", "displaced persons",
    "war economy", "wartime economy",
    "defence spending", "defense spending", "military spending",
    "occupation", "occupied territories",
    "ceasefire", "armistice",
    "battlefield", "frontline", "frontlines",
    "weapons", "armaments",
    "deployment", "troop deployment",
    "supply chain disruption", "supply-chain disruption",
})

_WAR_DOMAIN_ES = frozenset({
    "guerra", "militar", "conflicto armado",
    "terror", "terrorismo", "terrorista", "terroristas",
    "geopolítico", "geopolítica",
    "sanciones", "sanción", "embargo comercial", "embargo",
    "invasión", "invadir",
    "nuclear", "misil", "misiles",
    "conflicto", "conflictos", "tensiones",
    "movilización", "movilizar",
    "conscripción", "reclutamiento militar", "leva",
    "refugiado", "refugiados", "desplazados",
    "economía de guerra",
    "gasto militar", "gasto en defensa",
    "ocupación", "territorios ocupados",
    "alto el fuego", "armisticio",
    "campo de batalla", "frente",
    "armas", "armamento",
    "despliegue de tropas",
    "disrupción de la cadena de suministro",
})

_WAR_DOMAIN_PT = frozenset({
    "guerra", "militar", "conflito armado",
    "terror", "terrorismo", "terrorista", "terroristas",
    "geopolítico", "geopolítica",
    "sanções", "sanção", "embargo comercial", "embargo",
    "invasão", "invadir",
    "nuclear", "míssil", "mísseis",
    "conflito", "conflitos", "tensões",
    "mobilização", "mobilizar",
    "conscrição", "alistamento militar", "recrutamento militar",
    "refugiado", "refugiados", "deslocados",
    "economia de guerra",
    "gasto militar", "gasto com defesa",
    "ocupação", "territórios ocupados",
    "cessar-fogo", "armistício",
    "campo de batalha", "frente",
    "armas", "armamento",
    "deslocamento de tropas",
    "ruptura na cadeia de suprimentos",
})

_WAR_DOMAIN_DE = frozenset({
    "krieg", "militär", "bewaffneter konflikt",
    "terror", "terrorismus", "terrorist", "terroristen",
    "geopolitisch", "geopolitik",
    "sanktionen", "sanktion", "handelsembargo", "embargo",
    "invasion", "einmarsch",
    "nuklear", "rakete", "raketen",
    "konflikt", "konflikte", "spannungen",
    "mobilmachung", "mobilisierung",
    "wehrpflicht", "einberufung",
    "flüchtling", "flüchtlinge", "vertriebene",
    "kriegswirtschaft",
    "verteidigungsausgaben", "militärausgaben",
    "besatzung", "besetzte gebiete",
    "waffenstillstand",
    "schlachtfeld", "frontlinie",
    "waffen", "rüstung",
    "truppenstationierung",
    "lieferkettenstörung",
})

_WAR_DOMAIN_FR = frozenset({
    "guerre", "militaire", "conflit armé",
    "terreur", "terrorisme", "terroriste", "terroristes",
    "géopolitique",
    "sanctions", "sanction", "embargo commercial", "embargo",
    "invasion", "envahir",
    "nucléaire", "missile", "missiles",
    "conflit", "conflits", "tensions",
    "mobilisation", "mobiliser",
    "conscription", "service militaire", "appel sous les drapeaux",
    "réfugié", "réfugiés", "déplacés",
    "économie de guerre",
    "dépenses militaires", "dépenses de défense",
    "occupation", "territoires occupés",
    "cessez-le-feu", "armistice",
    "champ de bataille", "front",
    "armes", "armement",
    "déploiement de troupes",
    "perturbation de la chaîne d'approvisionnement",
})

_WAR_DOMAIN_IT = frozenset({
    "guerra", "militare", "conflitto armato",
    "terrore", "terrorismo", "terrorista", "terroristi",
    "geopolitico", "geopolitica",
    "sanzioni", "sanzione", "embargo commerciale", "embargo",
    "invasione", "invadere",
    "nucleare", "missile", "missili",
    "conflitto", "conflitti", "tensioni",
    "mobilitazione", "mobilitare",
    "coscrizione", "leva militare", "reclutamento militare",
    "rifugiato", "rifugiati", "sfollati",
    "economia di guerra",
    "spesa militare", "spese per la difesa",
    "occupazione militare", "territori occupati",
    "cessate il fuoco", "armistizio",
    "campo di battaglia", "fronte",
    "armi", "armamenti",
    "schieramento di truppe",
    "interruzione della catena di approvvigionamento",
})

_WAR_DOMAIN_JA = frozenset({
    "戦争", "戦時", "軍事", "武力紛争",
    "テロ", "テロリズム", "テロリスト",
    "地政学", "地政学的",
    "制裁", "経済制裁", "通商禁止", "禁輸",
    "侵攻", "侵略",
    "核", "ミサイル",
    "紛争", "対立", "緊張",
    "動員", "軍事動員",
    "徴兵", "徴兵制",
    "難民", "避難民",
    "戦時経済",
    "軍事支出", "防衛費",
    "占領", "占領地",
    "停戦", "休戦",
    "戦場", "前線",
    "兵器", "武器",
    "部隊展開",
    "サプライチェーンの混乱",
})

_WAR_DOMAIN_ZH = frozenset({
    "战争", "战时", "军事", "武装冲突",
    "恐怖", "恐怖主义", "恐怖分子",
    "地缘政治", "地缘政治的",
    "制裁", "经济制裁", "贸易禁运", "禁运",
    "入侵", "侵略",
    "核", "导弹",
    "冲突", "紧张局势", "对立",
    "动员", "军事动员",
    "征兵", "义务兵役",
    "难民", "流离失所者",
    "战时经济",
    "军费", "国防开支",
    "占领", "占领区",
    "停火", "休战",
    "战场", "前线",
    "武器", "军备",
    "部队部署",
    "供应链中断",
})


# ---------------------------------------------------------------------------
# Tech-labor UPSIDE / DOWNSIDE framing (Slice c).
# Used in 4-group co-occurrence to decompose LTUI into asymmetric
# narrative components:
#   labor × uncertainty × tech × {upside | downside}
# Seed examples (EN):
#   upside:   "productivity gains", "augmentation", "complement"
#   downside: "displace", "automate away", "obsolete"
# Reviewer log: puremacro/docs/lexicon_review.md
# ---------------------------------------------------------------------------
_TECH_LABOR_UPSIDE_EN = frozenset({
    "productivity gain", "productivity gains", "productivity boost",
    "productivity improvement",
    "augment", "augmented", "augmenting", "augmentation",
    "complement", "complementarity", "complementary",
    "new jobs", "job creation", "create jobs",
    "new occupation", "new occupations", "emerging occupations",
    "upskill", "upskilling", "reskill", "reskilling",
    "skill upgrade",
    "innovation-driven", "innovation driven",
    "ai adoption boost", "boost productivity",
    "complement workers", "human-ai collaboration",
    "labor-augmenting", "labour-augmenting",
    "wage premium",
})

_TECH_LABOR_UPSIDE_ES = frozenset({
    "ganancias de productividad", "aumento de productividad",
    "mejora de productividad",
    "aumentar", "aumentado", "aumento",
    "complementariedad", "complementario",
    "nuevos empleos", "creación de empleo",
    "nuevas ocupaciones",
    "recualificación", "actualización de habilidades",
    "mejora de competencias",
    "impulsado por la innovación",
    "complemento a los trabajadores",
    "colaboración humano-ia",
    "prima salarial",
})

_TECH_LABOR_UPSIDE_PT = frozenset({
    "ganhos de produtividade", "aumento de produtividade",
    "melhoria de produtividade",
    "aumentar", "aumentado", "aumento",
    "complementaridade", "complementar",
    "novos empregos", "criação de empregos",
    "novas ocupações",
    "requalificação", "atualização de habilidades",
    "melhoria de competências",
    "impulsionado pela inovação",
    "complemento aos trabalhadores",
    "colaboração humano-ia",
    "prêmio salarial",
})

_TECH_LABOR_UPSIDE_DE = frozenset({
    "produktivitätsgewinne", "produktivitätssteigerung",
    "produktivitätsverbesserung",
    "ergänzen", "ergänzend", "komplementarität",
    "neue arbeitsplätze", "arbeitsplatzschaffung",
    "neue berufe",
    "umschulung", "weiterbildung",
    "qualifikationsverbesserung",
    "innovationsgetrieben",
    "mensch-ki-zusammenarbeit",
    "lohnaufschlag",
})

_TECH_LABOR_UPSIDE_FR = frozenset({
    "gains de productivité", "amélioration de productivité",
    "augmentation de la productivité",
    "augmenter", "augmenté", "augmentation",
    "complémentarité", "complémentaire",
    "nouveaux emplois", "création d'emplois",
    "nouvelles professions",
    "requalification", "perfectionnement",
    "mise à niveau des compétences",
    "axé sur l'innovation",
    "collaboration humain-ia",
    "prime salariale",
})

_TECH_LABOR_UPSIDE_IT = frozenset({
    "guadagni di produttività", "aumento di produttività",
    "miglioramento della produttività",
    "aumentare", "aumentato", "aumento",
    "complementarità", "complementare",
    "nuovi posti di lavoro", "creazione di posti di lavoro",
    "nuove professioni",
    "riqualificazione", "aggiornamento delle competenze",
    "miglioramento delle competenze",
    "guidato dall'innovazione",
    "collaborazione uomo-ia",
    "premio salariale",
})

_TECH_LABOR_UPSIDE_JA = frozenset({
    "生産性向上", "生産性の向上", "生産性の改善",
    "拡張", "補完",
    "新規雇用", "雇用創出",
    "新たな職業",
    "再教育", "リスキリング", "スキルアップ",
    "イノベーション主導",
    "人間とaiの協働",
    "賃金プレミアム",
})

_TECH_LABOR_UPSIDE_ZH = frozenset({
    "生产率提升", "生产率提高", "生产率改善",
    "增强", "互补",
    "新岗位", "创造就业",
    "新兴职业",
    "再培训", "技能升级",
    "创新驱动",
    "人机协作", "人工智能协作",
    "工资溢价",
})

_TECH_LABOR_DOWNSIDE_EN = frozenset({
    "displace", "displaced", "displacement", "displacing",
    "replace workers", "replaces workers", "replacing workers",
    "automate away", "automated away", "automating away",
    "obsolete", "obsolescence",
    "redundant", "redundancy", "redundancies",
    "job loss", "job losses", "job destruction",
    "labor-saving", "labour-saving",
    "lay off", "layoff", "layoffs",
    "skill erosion",
    "downward pressure on wages", "wage pressure",
    "polarization", "polarisation",
    "hollowing out",
    "task displacement",
    "routine-biased",
})

_TECH_LABOR_DOWNSIDE_ES = frozenset({
    "desplazar", "desplazado", "desplazamiento",
    "reemplazar trabajadores", "sustituir trabajadores",
    "automatizar", "automatizado",
    "obsoleto", "obsolescencia",
    "redundante", "redundancia",
    "pérdida de empleo", "pérdida de empleos", "destrucción de empleo",
    "ahorro de mano de obra",
    "despidos", "despedir",
    "erosión de habilidades",
    "presión a la baja sobre los salarios",
    "polarización",
    "vaciamiento",
    "desplazamiento de tareas",
})

_TECH_LABOR_DOWNSIDE_PT = frozenset({
    "deslocar", "deslocado", "deslocamento",
    "substituir trabalhadores",
    "automatizar", "automatizado",
    "obsoleto", "obsolescência",
    "redundante", "redundância",
    "perda de emprego", "perda de empregos", "destruição de empregos",
    "economia de mão-de-obra",
    "demissões", "demitir",
    "erosão de habilidades",
    "pressão descendente sobre os salários",
    "polarização",
    "esvaziamento",
    "deslocamento de tarefas",
})

_TECH_LABOR_DOWNSIDE_DE = frozenset({
    "verdrängen", "verdrängung",
    "arbeiter ersetzen", "arbeitnehmer ersetzen",
    "automatisieren", "automatisiert",
    "obsolet", "veraltet",
    "redundant", "überflüssig",
    "arbeitsplatzverlust", "arbeitsplatzverluste",
    "arbeitsplatzabbau",
    "arbeitssparend",
    "entlassungen", "entlassen",
    "qualifikationsverlust",
    "lohndruck", "lohnabwärtsdruck",
    "polarisierung",
    "aushöhlung",
    "aufgabenverdrängung",
})

_TECH_LABOR_DOWNSIDE_FR = frozenset({
    "déplacer", "déplacement",
    "remplacer les travailleurs",
    "automatiser", "automatisé",
    "obsolète", "obsolescence",
    "redondant", "redondance",
    "perte d'emploi", "pertes d'emploi", "destruction d'emplois",
    "économie de main-d'œuvre",
    "licenciements", "licencier",
    "érosion des compétences",
    "pression à la baisse sur les salaires",
    "polarisation",
    "évidement",
    "déplacement de tâches",
})

_TECH_LABOR_DOWNSIDE_IT = frozenset({
    "spostare", "spostamento", "sostituire",
    "sostituire i lavoratori",
    "automatizzare", "automatizzato",
    "obsoleto", "obsolescenza",
    "ridondante", "ridondanza",
    "perdita di lavoro", "perdite di lavoro", "distruzione di posti di lavoro",
    "risparmio di lavoro",
    "licenziamenti", "licenziare",
    "erosione delle competenze",
    "pressione al ribasso sui salari",
    "polarizzazione",
    "svuotamento",
    "spostamento di compiti",
})

_TECH_LABOR_DOWNSIDE_JA = frozenset({
    "代替", "置き換え", "置換",
    "労働者の代替",
    "陳腐化", "時代遅れ",
    "余剰", "冗長",
    "雇用喪失", "雇用減少", "職の消失",
    "労働節約的",
    "解雇", "人員削減",
    "技能の陳腐化",
    "賃金下落圧力",
    "二極化",
    "中抜き",
    "業務代替",
})

_TECH_LABOR_DOWNSIDE_ZH = frozenset({
    "替代", "取代", "替换",
    "替代工人", "取代工人",
    "过时", "淘汰",
    "冗余", "多余",
    "失业", "工作流失", "工作岗位消失",
    "节省劳动力",
    "裁员", "解雇",
    "技能贬值",
    "工资下行压力",
    "两极分化",
    "中空化",
    "任务替代",
})


LEXICONS: dict = {
    "epu": {
        "en": _EPU_EN, "es": _EPU_ES, "pt": _EPU_PT,
        "de": _EPU_DE, "fr": _EPU_FR, "it": _EPU_IT,
        "ja": _EPU_JA, "zh": _EPU_ZH,
    },
    "mpu": {
        "en": _MPU_EN, "es": _MPU_ES, "pt": _MPU_PT,
        "de": _MPU_DE, "fr": _MPU_FR, "it": _MPU_IT,
        "ja": _MPU_JA, "zh": _MPU_ZH,
    },
    "gpr": {
        "en": _GPR_EN, "es": _GPR_ES, "pt": _GPR_PT,
        "de": _GPR_DE, "fr": _GPR_FR, "it": _GPR_IT,
        "ja": _GPR_JA, "zh": _GPR_ZH,
    },
    "tone": {
        "en": _TONE_EN, "es": _TONE_ES, "pt": _TONE_PT,
        "de": _TONE_DE, "fr": _TONE_FR, "it": _TONE_IT,
        "ja": _TONE_JA, "zh": _TONE_ZH,
    },
    "wui": {
        "en": _WUI_EN, "es": _WUI_ES, "pt": _WUI_PT,
        "de": _WUI_DE, "fr": _WUI_FR, "it": _WUI_IT,
        "ja": _WUI_JA, "zh": _WUI_ZH,
    },
    "lui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "phrases": _LUI_PHRASES_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "phrases": _LUI_PHRASES_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "phrases": _LUI_PHRASES_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "phrases": _LUI_PHRASES_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "phrases": _LUI_PHRASES_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "phrases": _LUI_PHRASES_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "phrases": _LUI_PHRASES_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "phrases": _LUI_PHRASES_ZH,
        },
    },
    "ltui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "tech_domain": _TECH_DOMAIN_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "tech_domain": _TECH_DOMAIN_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "tech_domain": _TECH_DOMAIN_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "tech_domain": _TECH_DOMAIN_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "tech_domain": _TECH_DOMAIN_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "tech_domain": _TECH_DOMAIN_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "tech_domain": _TECH_DOMAIN_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "tech_domain": _TECH_DOMAIN_ZH,
        },
    },
    "ltui_up": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "tech_domain": _TECH_DOMAIN_EN,
            "polarity": _TECH_LABOR_UPSIDE_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "tech_domain": _TECH_DOMAIN_ES,
            "polarity": _TECH_LABOR_UPSIDE_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "tech_domain": _TECH_DOMAIN_PT,
            "polarity": _TECH_LABOR_UPSIDE_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "tech_domain": _TECH_DOMAIN_DE,
            "polarity": _TECH_LABOR_UPSIDE_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "tech_domain": _TECH_DOMAIN_FR,
            "polarity": _TECH_LABOR_UPSIDE_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "tech_domain": _TECH_DOMAIN_IT,
            "polarity": _TECH_LABOR_UPSIDE_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "tech_domain": _TECH_DOMAIN_JA,
            "polarity": _TECH_LABOR_UPSIDE_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "tech_domain": _TECH_DOMAIN_ZH,
            "polarity": _TECH_LABOR_UPSIDE_ZH,
        },
    },
    "ltui_down": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "tech_domain": _TECH_DOMAIN_EN,
            "polarity": _TECH_LABOR_DOWNSIDE_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "tech_domain": _TECH_DOMAIN_ES,
            "polarity": _TECH_LABOR_DOWNSIDE_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "tech_domain": _TECH_DOMAIN_PT,
            "polarity": _TECH_LABOR_DOWNSIDE_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "tech_domain": _TECH_DOMAIN_DE,
            "polarity": _TECH_LABOR_DOWNSIDE_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "tech_domain": _TECH_DOMAIN_FR,
            "polarity": _TECH_LABOR_DOWNSIDE_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "tech_domain": _TECH_DOMAIN_IT,
            "polarity": _TECH_LABOR_DOWNSIDE_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "tech_domain": _TECH_DOMAIN_JA,
            "polarity": _TECH_LABOR_DOWNSIDE_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "tech_domain": _TECH_DOMAIN_ZH,
            "polarity": _TECH_LABOR_DOWNSIDE_ZH,
        },
    },
    "lwui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "war_domain": _WAR_DOMAIN_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "war_domain": _WAR_DOMAIN_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "war_domain": _WAR_DOMAIN_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "war_domain": _WAR_DOMAIN_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "war_domain": _WAR_DOMAIN_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "war_domain": _WAR_DOMAIN_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "war_domain": _WAR_DOMAIN_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "war_domain": _WAR_DOMAIN_ZH,
        },
    },
    "lwui_wage": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "wage_domain": _WAGE_DOMAIN_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "wage_domain": _WAGE_DOMAIN_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "wage_domain": _WAGE_DOMAIN_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "wage_domain": _WAGE_DOMAIN_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "wage_domain": _WAGE_DOMAIN_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "wage_domain": _WAGE_DOMAIN_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "wage_domain": _WAGE_DOMAIN_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "wage_domain": _WAGE_DOMAIN_ZH,
        },
    },
}


__all__ = ["LEXICONS"]
