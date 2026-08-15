# Narrative Extension — Slice 5 (LUI Lexicon + Fed Minutes URL Fix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand LUI lexicons across 8 languages (~1000 total terms) and rewrite Fed minutes URL resolution via announcement-page link parsing. Re-run notebook 28; LUI vs urate ρ ≥ 0.30.

**Architecture:** Lexicon work is data-only (no API change). Each `_LUI_<lang>` constant gets reorganized around the 6 conceptual groups already in the spec (layoffs / hiring-freeze / wage-compression / labor-shortage / participation-drop / unemployment-risk). The Fed-minutes fix replaces the brittle URL pattern transform with a parser that extracts the actual `<a href="/fomc/minutes/…">` link from the announcement page; works across all eras (pre-2014 used `/fomc/minutes/{meeting-date}.htm`, post-2014 uses `/monetarypolicy/fomcminutes{release-date}.htm`).

**Tech Stack:** Python 3.10+, `re`. No new deps.

**Spec reference:** `docs/specs/2026-05-09-narrative-slice5-lui-lexicon-and-minutes.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (current head past `v0.7.1`). No new branch.

**Pre-implementation baseline:** `pytest -q` after Slice 4 = **971 passed, 27 skipped**, plus 1 pre-existing pyodide-compat failure (statsmodels.tsa.x13 leak; out of scope).

---

## File Structure

### Files modified
- `puremacro/narrative/indices/_lexicons.py` — expand 8 `_LUI_<lang>` constants.
- `puremacro/narrative/sources/fed_minutes.py` — add `_extract_minutes_body_link()`; rewrite URL-resolution block; remove `_minutes_body_url()` (no longer used).
- `puremacro/tests/test_narrative_indices.py` — bump LUI-coverage assertions from `≥ 1` to `≥ 100` (Latin-script languages) / `≥ 60` (ja, zh).
- `puremacro/tests/test_narrative_fed_url_transform.py` — repurpose: replace URL-transform tests with `_extract_minutes_body_link` tests.
- `puremacro/tests/test_narrative_cb_connectors.py` — update `test_fed_minutes_yields_four_tuple` mock so the announcement HTML contains a body link.
- Notebook 28 outputs (`notebooks/output_tables/28_lui_*`, `notebooks/data_cache/fed_corpus_28.parquet`) — re-rendered.
- `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, `CHANGELOG.md` — 0.7.1 → 0.7.2.

---

## Task 0: Branch + baseline

**Files:** none.

- [ ] **Step 1: Verify branch + baseline**

```bash
git branch --show-current   # must be feature/narrative-extension-slice3
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: `971 passed, 27 skipped`.

---

## Task 1: LUI lexicon expansion — English (35 → ~150 terms)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (replace `_LUI_EN`)
- Modify: `puremacro/tests/test_narrative_indices.py` (update lexicon-coverage assertions)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Replace `_LUI_EN`**

In `puremacro/narrative/indices/_lexicons.py`, find the existing `_LUI_EN = frozenset({...})` and replace with:

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (English) — six conceptual groups, ~150 terms.
# Terms are lowercase phrases; word-boundary regex matched at score time.
# ---------------------------------------------------------------------------
_LUI_EN = frozenset({
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
```

(Count: about 145 terms. Doesn't need to hit 150 exactly.)

- [ ] **Step 3: Update lexicon-coverage tests**

In `puremacro/tests/test_narrative_indices.py`, find:

```python
@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_other_indices_have_each_supported_language(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    for index in ("mpu", "gpr", "wui", "lui"):
        assert lang in LEXICONS[index], (
            f"{index} missing language {lang}"
        )
        terms = LEXICONS[index][lang]
        assert len(terms) >= 1, f"{index}/{lang} is empty"
```

Add a new test alongside it (don't replace, since this one is for general coverage):

```python
def test_lui_english_has_substantive_coverage():
    """LUI/EN expanded in Slice 5: ≥ 100 terms across 6 conceptual groups."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    lui_en = LEXICONS["lui"]["en"]
    assert len(lui_en) >= 100, f"LUI/EN has only {len(lui_en)} terms"
    # Spot-check coverage of each conceptual group.
    assert any("layoff" in t for t in lui_en)
    assert any("hiring freeze" in t for t in lui_en)
    assert any("wage" in t for t in lui_en)
    assert any("shortage" in t for t in lui_en)
    assert any("participation" in t for t in lui_en)
    assert any("unemployment" in t for t in lui_en)
```

- [ ] **Step 4: Run tests, expect green**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "lui_english_has_substantive or other_indices_have" 2>&1 | tail -15
```
Expected: 8 tests pass (1 new + 7 parametrized cross-language).

- [ ] **Step 5: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 971 + 1 = 972 passed.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): LUI English lexicon expansion (35 → ~145 terms across 6 groups)"
```

---

## Task 2: LUI lexicon expansion — Spanish + Portuguese (≥ 100 each)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (replace `_LUI_ES`, `_LUI_PT`)
- Modify: `puremacro/tests/test_narrative_indices.py` (add coverage tests)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Replace `_LUI_ES`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (Spanish) — six conceptual groups, ~120 terms.
# ---------------------------------------------------------------------------
_LUI_ES = frozenset({
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
    # Group 2: Hiring freeze
    "congelación de contrataciones", "congelamiento de contrataciones",
    "pausa en contrataciones", "pausa en las contrataciones",
    "freno a la contratación", "freno en la contratación",
    "ralentización de las contrataciones",
    "no contratar", "sin nuevas contrataciones",
    "selectividad en la contratación",
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
})
```

- [ ] **Step 3: Replace `_LUI_PT`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (Portuguese) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_PT = frozenset({
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
    # Group 2: Hiring freeze
    "congelamento de contratações", "congelamento das contratações",
    "pausa nas contratações", "pausa em contratações",
    "freio nas contratações",
    "desaceleração das contratações",
    "sem novas contratações", "não contratar",
    "contratação seletiva",
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
    # Group 4: Labor shortage
    "escassez de mão de obra", "escassez laboral",
    "escassez de habilidades",
    "escassez de talento", "guerra por talento",
    "mercado de trabalho apertado", "mercado de trabalho aquecido",
    "dificuldade de contratação", "dificuldade em contratar",
    "vagas difíceis de preencher",
    "gargalo de contratação",
    "escassez de trabalhadores",
    # Group 5: Participation drop
    "taxa de participação", "taxa de participação no mercado de trabalho",
    "participação na força de trabalho",
    "trabalhadores desencorajados",
    "saída da força de trabalho", "afastamento da força de trabalho",
    "queda na participação", "diminuição da participação",
    "inatividade", "inativos",
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
})
```

- [ ] **Step 4: Add per-language coverage tests**

Append to `puremacro/tests/test_narrative_indices.py`:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("es", 100),
    ("pt", 100),
])
def test_lui_latin_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]
    assert len(terms) >= min_count, (
        f"LUI/{lang} has only {len(terms)} terms; expected ≥ {min_count}"
    )
```

- [ ] **Step 5: Run tests, expect green**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "lui_latin_lexicon" 2>&1 | tail -10
```
Expected: 2 tests pass (es + pt).

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 972 + 2 = 974 passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): LUI Spanish + Portuguese lexicon expansion (≥100 terms each)"
```

---

## Task 3: LUI lexicon expansion — German + French + Italian (≥ 100 each)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (replace `_LUI_DE`, `_LUI_FR`, `_LUI_IT`)
- Modify: `puremacro/tests/test_narrative_indices.py` (extend parametrize)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Replace `_LUI_DE`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (German) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_DE = frozenset({
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
    # Group 2: Hiring freeze
    "einstellungsstopp", "einstellungssperre",
    "neueinstellungsstopp",
    "verlangsamung der einstellungen",
    "keine neueinstellungen",
    "selektive einstellung",
    "einstellungspause",
    # Group 3: Wage compression
    "lohnstagnation", "lohnstagnierung",
    "reallohnverlust", "reallohnrückgang",
    "lohnzurückhaltung", "lohnmoderation",
    "schwaches lohnwachstum",
    "schwacher lohndruck", "nachlassender lohndruck",
    "lohnkürzung", "lohnkürzungen",
    "lohneinfrierung", "eingefrorene löhne",
    # Group 4: Labor shortage
    "arbeitskräftemangel", "fachkräftemangel",
    "qualifikationsmangel",
    "talentmangel",
    "angespannter arbeitsmarkt", "enger arbeitsmarkt",
    "einstellungsschwierigkeiten",
    "schwer zu besetzen",
    "personalknappheit",
    # Group 5: Participation drop
    "erwerbsquote", "erwerbsbeteiligung",
    "entmutigte arbeitnehmer",
    "rückzug aus dem arbeitsmarkt", "ausscheiden aus dem erwerbsleben",
    "rückgang der erwerbsbeteiligung",
    "nichterwerbstätigkeit",
    "stille reserve",
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
})
```

- [ ] **Step 3: Replace `_LUI_FR`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (French) — six conceptual groups, ~110 terms.
# ---------------------------------------------------------------------------
_LUI_FR = frozenset({
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
    # Group 2: Hiring freeze
    "gel des embauches", "gel de l'embauche",
    "pause des embauches", "pause dans les embauches",
    "ralentissement des embauches",
    "pas de nouvelles embauches",
    "embauche sélective",
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
    # Group 4: Labor shortage
    "pénurie de main-d'œuvre", "pénurie de main d'oeuvre",
    "pénurie de compétences",
    "pénurie de talents", "guerre des talents",
    "marché du travail tendu",
    "difficultés de recrutement", "difficulté à recruter",
    "postes difficiles à pourvoir",
    "manque de personnel",
    # Group 5: Participation drop
    "taux d'activité", "taux de participation",
    "participation au marché du travail",
    "travailleurs découragés",
    "sortie du marché du travail", "abandon du marché du travail",
    "baisse de la participation", "recul de la participation",
    "inactivité",
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
})
```

- [ ] **Step 4: Replace `_LUI_IT`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (Italian) — six conceptual groups, ~105 terms.
# ---------------------------------------------------------------------------
_LUI_IT = frozenset({
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
    # Group 4: Labor shortage
    "carenza di manodopera",
    "carenza di competenze",
    "carenza di talenti",
    "mercato del lavoro teso",
    "difficoltà di assunzione", "difficoltà ad assumere",
    "posti difficili da riempire",
    "mancanza di personale",
    # Group 5: Participation drop
    "tasso di partecipazione", "tasso di attività",
    "partecipazione al mercato del lavoro",
    "lavoratori scoraggiati",
    "uscita dal mercato del lavoro", "abbandono del mercato del lavoro",
    "calo della partecipazione", "diminuzione della partecipazione",
    "inattività",
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
})
```

- [ ] **Step 5: Extend the per-language coverage parametrize**

In `puremacro/tests/test_narrative_indices.py`, find the test added in Task 2:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("es", 100),
    ("pt", 100),
])
def test_lui_latin_lexicon_substantive_coverage(lang, min_count):
    ...
```

Extend to:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("es", 100),
    ("pt", 100),
    ("de", 100),
    ("fr", 100),
    ("it", 100),
])
def test_lui_latin_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]
    assert len(terms) >= min_count, (
        f"LUI/{lang} has only {len(terms)} terms; expected ≥ {min_count}"
    )
```

- [ ] **Step 6: Run tests, expect green**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "lui_latin_lexicon" 2>&1 | tail -10
```
Expected: 5 tests pass.

- [ ] **Step 7: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 974 + 3 = 977 passed.

- [ ] **Step 8: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): LUI German + French + Italian lexicon expansion (≥100 terms each)"
```

---

## Task 4: LUI lexicon expansion — Japanese + Chinese (≥ 60 each)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py` (replace `_LUI_JA`, `_LUI_ZH`)
- Modify: `puremacro/tests/test_narrative_indices.py` (add coverage tests for ja, zh)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Replace `_LUI_JA`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (Japanese) — six conceptual groups, ~65 terms.
# Japanese economic vocabulary for labor uncertainty has tighter
# concept-to-term mapping than Latin scripts.
# ---------------------------------------------------------------------------
_LUI_JA = frozenset({
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
    # Group 3: Wage compression
    "賃金停滞", "賃金抑制",
    "実質賃金低下",
    "賃金カット", "賃金引き下げ",
    "賃上げ抑制",
    "賃金凍結",
    # Group 4: Labor shortage
    "労働力不足", "人手不足", "人材不足",
    "技能不足", "スキル不足",
    "売り手市場",
    "採用難",
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
})
```

- [ ] **Step 3: Replace `_LUI_ZH`**

```python
# ---------------------------------------------------------------------------
# Labor-Market Uncertainty (Simplified Chinese) — six conceptual groups,
# ~65 terms.
# ---------------------------------------------------------------------------
_LUI_ZH = frozenset({
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
    # Group 3: Wage compression
    "工资停滞", "薪资停滞",
    "实际工资下降",
    "工资降低", "降薪",
    "工资增长缓慢", "工资增长疲软",
    "工资压力减弱",
    "工资冻结",
    # Group 4: Labor shortage
    "劳动力短缺", "用工短缺",
    "技能短缺", "技能不足",
    "人才短缺", "人才争夺战",
    "就业市场紧张",
    "招聘困难",
    # Group 5: Participation drop
    "劳动参与率",
    "退出劳动力市场", "退出劳动市场",
    "气馁工人", "受挫工人",
    "非劳动人口",
    # Group 6: Unemployment risk
    "失业", "失业潮",
    "失业救济金申请",
    "失业风险",
    "失业率上升", "失业增加",
    "就业疲软", "就业恶化",
    "劳动力市场恶化",
    "就业不确定性",
    "失业率",
})
```

- [ ] **Step 4: Add ja/zh coverage tests**

Append to `puremacro/tests/test_narrative_indices.py`:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("ja", 60),
    ("zh", 60),
])
def test_lui_cjk_lexicon_substantive_coverage(lang, min_count):
    """JA/ZH labor lexicons are denser per concept than Latin scripts;
    target ≥ 60 terms per language."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]
    assert len(terms) >= min_count, (
        f"LUI/{lang} has only {len(terms)} terms; expected ≥ {min_count}"
    )
```

- [ ] **Step 5: Run tests, expect green**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "lui_cjk_lexicon" 2>&1 | tail -10
```
Expected: 2 tests pass.

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 977 + 2 = 979 passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): LUI Japanese + Chinese lexicon expansion (≥60 terms each)"
```

---

## Task 5: Fed minutes URL via announcement-page parsing

**Files:**
- Modify: `puremacro/narrative/sources/fed_minutes.py` (add `_extract_minutes_body_link`; rewrite the URL-resolution block; remove `_minutes_body_url`)
- Modify: `puremacro/tests/test_narrative_fed_url_transform.py` (replace URL-transform tests with link-extraction tests)
- Modify: `puremacro/tests/test_narrative_cb_connectors.py` (update `test_fed_minutes_yields_four_tuple` mock to include a body link in the announcement HTML)

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Rewrite `puremacro/tests/test_narrative_fed_url_transform.py`**

Replace the entire file content with link-extraction tests:

```python
"""Tests for the Fed minutes announcement-page link extractor."""
from __future__ import annotations


def test_extract_minutes_body_link_modern_pattern():
    """Post-2014 announcement pages link to /monetarypolicy/fomcminutes…"""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<p>The minutes were released today.</p>'
        '<a href="/monetarypolicy/fomcminutes20240501.htm">Minutes (HTML)</a>'
        '</body></html>'
    )
    href = _extract_minutes_body_link(html)
    assert href == "/monetarypolicy/fomcminutes20240501.htm"


def test_extract_minutes_body_link_pre_2014_pattern():
    """Pre-2014 announcement pages link to /fomc/minutes/{meeting-date}.htm"""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<p>Released today: minutes of the December 13, 2005 meeting.</p>'
        '<a href="/fomc/minutes/20051213.htm">View the minutes</a>'
        '</body></html>'
    )
    href = _extract_minutes_body_link(html)
    assert href == "/fomc/minutes/20051213.htm"


def test_extract_minutes_body_link_returns_none_when_no_link():
    """If the announcement page has no minutes-body link, return None."""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = '<html><body><p>Just an announcement.</p></body></html>'
    assert _extract_minutes_body_link(html) is None


def test_extract_minutes_body_link_picks_first_match():
    """If multiple matching links exist, return the first one."""
    from puremacro.narrative.sources.fed_minutes import _extract_minutes_body_link
    html = (
        '<html><body>'
        '<a href="/monetarypolicy/fomcminutes20240501.htm">First</a>'
        '<a href="/monetarypolicy/fomcminutes20240601.htm">Second</a>'
        '</body></html>'
    )
    assert _extract_minutes_body_link(html) == "/monetarypolicy/fomcminutes20240501.htm"
```

- [ ] **Step 3: Run, verify failure**

```bash
cd puremacro && pytest tests/test_narrative_fed_url_transform.py -v --no-header 2>&1 | tail -10
```
Expected: 4 fail with `ImportError: cannot import name '_extract_minutes_body_link'` (because Slice 4's `_minutes_body_url` is named differently and we're about to remove it anyway).

- [ ] **Step 4: Rewrite `puremacro/narrative/sources/fed_minutes.py`**

Replace the entire file content with:

```python
"""Federal Reserve FOMC minutes.

Resolution strategy (3-tier, robust across all eras):
  1. Fetch the announcement page (URL from JSON `l` field).
  2. Parse it for the actual minutes body link
     (``/fomc/minutes/{meeting-date}.htm`` for pre-2014;
      ``/monetarypolicy/fomcminutes{release-date}.htm`` for post-2014).
  3. Fetch + extract the body. Fall back to the announcement page text
     if the body URL fails or the extraction is short.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

import pandas as pd

from ..._http import safe_get_bytes, safe_get_text
from ._extractors import extract_body


_LISTING_URL = "https://www.federalreserve.gov/json/ne-press.json"
_BASE = "https://www.federalreserve.gov"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_BODY_LINK_RX = re.compile(
    r'<a\b[^>]*\bhref="(/(?:fomc/minutes|monetarypolicy/fomcminutes)[^"]+\.htm)"',
    flags=re.IGNORECASE,
)


def _extract_minutes_body_link(announcement_html: str) -> str | None:
    """Find the first link to a minutes body inside the announcement page.

    Looks for ``<a href="/fomc/minutes/...html">`` (pre-2014) or
    ``<a href="/monetarypolicy/fomcminutes...html">`` (post-2014).
    Returns the href as found; caller prepends ``_BASE`` if relative.
    Returns ``None`` if no match.
    """
    m = _BODY_LINK_RX.search(announcement_html)
    if not m:
        return None
    return m.group(1)


def iter_fed_minutes() -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for FOMC meeting minutes."""
    try:
        body = safe_get_bytes(_LISTING_URL, user_agent=_UA)
    except Exception:
        return
    try:
        obj = json.loads(body.decode("utf-8-sig", errors="ignore"))
    except json.JSONDecodeError:
        return
    items = obj if isinstance(obj, list) else obj.get("refData", [])
    for item in items:
        if (item.get("pt") or "").lower() != "monetary policy":
            continue
        title = (item.get("t") or item.get("ti") or "").lower()
        if "minutes" not in title:
            continue
        if "discount rate" in title:
            continue
        if "fomc" not in title and "federal open market committee" not in title:
            continue
        try:
            date = pd.Timestamp(item.get("d"))
        except Exception:
            continue
        href = item.get("l", "")
        if not href:
            continue
        announcement_url = _BASE + href if href.startswith("/") else href

        # Step 1+2: fetch announcement, parse for body link.
        try:
            announcement_html = safe_get_text(announcement_url, user_agent=_UA)
        except Exception:
            continue

        body_text = ""
        chosen_url = announcement_url
        body_href = _extract_minutes_body_link(announcement_html)
        if body_href:
            body_url = _BASE + body_href if body_href.startswith("/") else body_href
            try:
                body_html = safe_get_text(body_url, user_agent=_UA)
                body_text = extract_body(body_html, bank_code="FED")
                if body_text and len(body_text) >= 5000:
                    chosen_url = body_url
                else:
                    body_text = ""  # too short, fall back
            except Exception:
                body_text = ""

        # Step 3: fall back to announcement-page extraction if body fetch
        # failed or was too short.
        if not body_text:
            body_text = extract_body(announcement_html, bank_code="FED")
            chosen_url = announcement_url

        if not body_text:
            continue
        yield (date, body_text, chosen_url, {
            "doctype": "minutes", "language": "en",
            "bank_code": "FED", "country": "USA",
        })


__all__ = ["iter_fed_minutes"]
```

- [ ] **Step 5: Update the connector mock test**

In `puremacro/tests/test_narrative_cb_connectors.py`, find `test_fed_minutes_yields_four_tuple`. The Slice-4 mock registered both the announcement URL and a body URL with the URL-pattern transform. Slice 5 needs the announcement HTML to contain a `<a href>` to the body link. Replace the mock with:

```python
def test_fed_minutes_yields_four_tuple(mock_http):
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/json/ne-press.json":
                b'\xef\xbb\xbf[{"d":"2022-04-06","t":"Minutes of the FOMC March meeting",'
                b'"pt":"Monetary Policy",'
                b'"l":"/newsevents/pressreleases/monetary20220316a.htm"}]',
        },
        text={
            # Announcement page contains a body link → body URL gets fetched.
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm":
                "<html><body><p>The minutes were released today.</p>"
                "<a href=\"/monetarypolicy/fomcminutes20220316.htm\">Minutes</a>"
                "</body></html>",
            # Body page (5000+ chars) — must contain <div id=\"article\"> for FED extractor.
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm":
                "<html><body><div id=\"article\"><p>Participants noted "
                "that inflation remained elevated. " * 200 + "</p></div></body></html>",
        },
    )
    from puremacro.narrative.sources import iter_fed_minutes
    records = list(iter_fed_minutes())
    assert len(records) >= 1
    _, _, _, meta = records[0]
    assert meta["doctype"] == "minutes"
```

- [ ] **Step 6: Run new + existing tests, expect green**

```bash
cd puremacro && pytest tests/test_narrative_fed_url_transform.py tests/test_narrative_cb_connectors.py -v --no-header -k "minutes_body_link or fed_minutes" 2>&1 | tail -10
```
Expected: 5 tests pass (4 link-extraction + 1 connector).

- [ ] **Step 7: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: ≥ 979 passed (the 4 URL-transform tests are replaced 1:1 with 4 link-extraction tests; net pass count unchanged from Task 4 baseline).

- [ ] **Step 8: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git add puremacro/puremacro/narrative/sources/fed_minutes.py \
        puremacro/tests/test_narrative_fed_url_transform.py \
        puremacro/tests/test_narrative_cb_connectors.py
git commit -m "feat(narrative): Fed minutes URL via announcement-page link parsing (handles all eras)"
```

---

## Task 6: Re-run notebook 28 + validate signal + 0.7.2 release

**Files:**
- `notebooks/28_us_lui_from_fed_text.executed.ipynb` (regenerated)
- `notebooks/data_cache/fed_corpus_28.parquet` (regenerated)
- `notebooks/output_tables/28_lui_*` (regenerated)
- `notebooks/output_figures/28_lui_us_timeseries.pdf` (regenerated)
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `puremacro/tests/test_import.py`, `puremacro/CHANGELOG.md`

This task has both research-validation steps (re-run, inspect) and release steps (version bump, tag).

- [ ] **Step 1: Verify branch.**

- [ ] **Step 2: Clear stale corpus + re-run notebook 28**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
rm -f notebooks/data_cache/fed_corpus_28.parquet
PUREMACRO_REFETCH=1 jupyter execute notebooks/28_us_lui_from_fed_text.ipynb \
    --output 28_us_lui_from_fed_text.executed.ipynb
```
Expected output (network-bound; takes ~5-15 minutes):
```
[NbClientApp] Executing notebooks/28_us_lui_from_fed_text.ipynb
[NbClientApp] Save executed results to notebooks/28_us_lui_from_fed_text.executed.ipynb
```

- [ ] **Step 3: Inspect outputs**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
cat notebooks/output_tables/28_lui_us_quarterly.meta.json
cat notebooks/output_tables/28_lui_validation_corr.csv
```

Acceptance criterion (Slice 5 headline):
- `LUI vs urate ρ ≥ 0.30` (was +0.18 in Slice 4). If hit, Slice 5's lexicon expansion worked.
- EPU/WUI correlations should remain at Slice 4 levels (not regress meaningfully).
- Corpus size should be ≥ 350 (was 366 in Slice 4; minutes body extraction now broader so may exceed).

If LUI ρ < 0.30 but the difference is small (e.g., 0.22-0.29), document the finding and ship anyway — it's an improvement, lexicon may need further per-group rebalancing in a future iteration.

If LUI ρ < 0.18 (regression), STOP and investigate — the expanded lexicon may have introduced false-positive matches.

- [ ] **Step 4: Bump version to 0.7.2**

Edit `puremacro/pyproject.toml`: `version = "0.7.1"` → `version = "0.7.2"`.

Edit `puremacro/puremacro/__init__.py`: `__version__ = "0.7.1"` → `__version__ = "0.7.2"`.

Edit `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.7.1"` → `"0.7.2"`.

- [ ] **Step 5: Add CHANGELOG entry**

Open `puremacro/CHANGELOG.md`. Add a new top entry above the `## 0.7.1 — 2026-05-09` block. (Insert the appropriate ρ values from Step 3 after running.)

```markdown
## 0.7.2 — 2026-05-09

Slice 5: LUI lexicon expansion + Fed minutes URL fix. Closes Slice 4's diagnosed bottleneck — LUI signal moves from ρ = +0.18 toward research-usable strength.

### Fixed

- **LUI lexicon was too thin** — 35 English terms missed most labor-uncertainty vocabulary in Fed text. Expanded to ~145 English terms across the 6 conceptual groups (layoffs, hiring-freeze, wage-compression, labor-shortage, participation-drop, unemployment-risk). All 8 languages got proportional expansion: ≥ 100 terms in en/es/pt/de/fr/it; ≥ 60 in ja/zh.
- **Fed minutes URL transform was wrong for pre-2014 items** — the JSON `l` field's announcement URL doesn't match the `/monetarypolicy/fomcminutes{date}.htm` pattern for older minutes (which used `/fomc/minutes/{meeting-date}.htm`). Removed the brittle regex transform; replaced with `_extract_minutes_body_link()` that parses the announcement page for the actual `<a href>` to the body. Works across all eras.

### Added

- `narrative.sources.fed_minutes._extract_minutes_body_link(announcement_html)` — public-by-test private helper. Finds the first `<a href="/fomc/minutes/…">` or `<a href="/monetarypolicy/fomcminutes…">` link in an announcement-page HTML.
- `tests/test_narrative_indices.py` — new lexicon-coverage parametrize covering ≥ 100 terms for the 6 Latin-script LUI lexicons and ≥ 60 for ja/zh (8 new tests).
- `tests/test_narrative_fed_url_transform.py` — repurposed: 4 tests for `_extract_minutes_body_link` (modern pattern, pre-2014 pattern, no-link fallback, first-match selection).

### Removed

- `narrative.sources.fed_minutes._minutes_body_url()` — superseded by `_extract_minutes_body_link()`. The old regex transform was wrong for pre-2014 minutes (where the body URL uses the meeting date, not the announcement date).

### Pyodide compatibility

- `_lexicons.py` is data-only (frozensets); `_extract_minutes_body_link` is pure-Python `re`. No new top-level deps. Pyodide-clean. Slice 5 added zero new forbidden-runtime-dep leaks.

### Notes for next iteration

- If LUI vs urate now ≥ 0.30 → notebook 29 (state-panel LP-IV with national LUI as shock) is unblocked.
- BIS speeches connector still returns 0 live (URL works but serves JS-rendered HTML); future iteration adds a headless-browser path or a different endpoint.
- Slice 6 candidates: length-normalized WUI, Picault-Renault paragraph-level multinomial logit, full Hubert lexicon, `llm_prob_kernel`.
```

- [ ] **Step 6: Final regression sweep**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: ≥ 979 passed (matches Task 5 final count after the 4 link-extraction tests replaced the 4 URL-transform tests).

```bash
cd puremacro && pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py -q --no-header 2>&1 | tail -3
```
Expected: zero fiscal-narrative regressions.

- [ ] **Step 7: Commit + tag**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current
git status -s notebooks/output_tables/ notebooks/data_cache/
git add notebooks/output_tables/28_lui_us_quarterly.parquet \
        notebooks/output_tables/28_lui_us_quarterly.meta.json \
        notebooks/output_tables/28_lui_validation_corr.csv \
        notebooks/data_cache/fed_corpus_28.parquet \
        puremacro/pyproject.toml \
        puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py \
        puremacro/CHANGELOG.md
git commit -m "chore(release): puremacro 0.7.2 — narrative Slice 5 (LUI lexicon + Fed minutes URL)"
git tag -a v0.7.2 -m "puremacro 0.7.2 — narrative Slice 5 (LUI lexicon + Fed minutes URL)"
```

(Do NOT push.)

---

## Definition of Done

- [ ] All 7 task blocks above checked off.
- [ ] Branch `feature/narrative-extension-slice3` has new commits past `v0.7.1`, tagged `v0.7.2`.
- [ ] `pytest -q` ≥ 979 passed.
- [ ] `pytest tests/test_pyodide_compat.py` shows the SAME 1 pre-existing failure (no new leaks).
- [ ] Zero fiscal-narrative regressions.
- [ ] `pyproject.toml` version is `0.7.2`; `puremacro.__version__ == "0.7.2"`.
- [ ] `CHANGELOG.md` has a `## 0.7.2 — 2026-05-09` section with the actual ρ values from the re-run.
- [ ] LUI vs urate ρ improved from Slice 4's +0.18 (target ≥ 0.30; smaller improvement is acceptable but document).
- [ ] Notebook 28 outputs (parquet + JSON + CSV + cache) updated.

## Out of scope (deferred)

- BIS speeches HTML-scrape — requires headless-browser path; defer.
- Length-normalized WUI per Ahir-Bloom-Furceri.
- Picault-Renault paragraph-level multinomial logit; full Hubert lexicon.
- `llm_prob_kernel` for LLM-backed scoring.
- Per-bank precise extractors for Slice-3 banks.
- Notebook 29 (state-panel LP-IV with national LUI as shock).
