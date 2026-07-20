> 🇬🇧 English · 🇪🇸 [Español](es/lexicon_review.md)

# Lexicon Reviewer Log

This log records who reviewed each language lexicon, against which anchor
documents, and on what date. Required for any `_<DOMAIN>_<lang>` lexicon
shipped under `puremacro.narrative.indices._lexicons`.

## `_TECH_DOMAIN`

| Language | Reviewer | Anchor docs (≥ 3) | Date |
|---|---|---|---|
| en | Jorge Alonso Ortiz | BBD AI-EPU 2023 paper §2; Felten-Raj-Seamans 2021 SMJ Tables 1–2; 3 FOMC speeches Powell (2023-05, 2024-01, 2024-06) referencing AI | 2026-05-11 |
| es | Jorge Alonso Ortiz | Banxico Informe Trimestral 2024-Q1 §III; BCRA Informe de Política Monetaria 2024-03; CEPAL "AI en el mercado laboral latinoamericano" 2023 | 2026-05-11 |
| pt | Jorge Alonso Ortiz | BCB COPOM Ata 2024-09 (minutes 264); BCB Relatório de Inflação 2024-Q2 §3; Discurso Roberto Campos Neto 2024-04 (digitalização) | 2026-05-11 |
| de | Jorge Alonso Ortiz | Bundesbank Monatsbericht 2024-04 (KI-Wirtschaft); ECB Lagarde speech 2023-10-25 (AI / labour); Bundesbank Speech Mauderer 2024-03 | 2026-05-11 |
| fr | Jorge Alonso Ortiz | Banque de France Bulletin 2024-Q1 §2 (numérique); ECB Lagarde 2024 Sintra opening remarks; Trésor Trésor-Éco n°337 (mai 2024) | 2026-05-11 |
| it | Jorge Alonso Ortiz | Banca d'Italia Relazione Annuale 2023 §III (digitalizzazione); Banca d'Italia speech Visco 2024-02; ISTAT Rapporto Annuale 2024 §3 | 2026-05-11 |
| ja | Jorge Alonso Ortiz | BoJ Outlook Report 2024-10 (生成AI section); BoJ speech Ueda 2024-05; Cabinet Office White Paper on the Economy 2024 | 2026-05-11 |
| zh | Jorge Alonso Ortiz | PBoC 2024Q1 Monetary Policy Report §专栏; PBoC Pan Gongsheng 2024 speech (人工智能); China Statistical Yearbook 2024 ch.20 | 2026-05-11 |

## 2026-05-12 — α.2 stability variants

**Reviewer:** Jorge Alonso Ortiz
**Scope:** `_*_EXPANDED` lexicon variants for the 5 in-scope indices
(`lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`) × 8 languages. Six
families touched — `_LABOR_DOMAIN`, `_UNCERTAINTY_TONE`, `_TECH_DOMAIN`,
`_TECH_LABOR_UPSIDE`, `_TECH_LABOR_DOWNSIDE`, `_WAR_DOMAIN` — for a
total of 48 distinct `_EXPANDED` frozensets (plus 8 `_LUI_PHRASES_*`
passthroughs reused verbatim).

Each expansion adds ≤ `floor(0.20 × |base|)` terms (cap enforced by
`puremacro/tests/test_narrative_lexicons_expanded.py`), sourced from a
per-family English seed list and translated into each target language:

- `_UNCERTAINTY_TONE_*`: subdued / muted / cloudy / wary / hesitant.
- `_LABOR_DOMAIN_*`: workforce / staffing / payrolls / labour-supply
  trend / manpower.
- `_TECH_DOMAIN_*`: large language model / generative model / AI
  assistant / digitisation / automation wave.
- `_TECH_LABOR_UPSIDE_*`: skill upgrade / productivity boost / new
  occupations / human–AI complementarity.
- `_TECH_LABOR_DOWNSIDE_*`: labour-saving / obsolete skills / redundant
  roles / AI replacement.
- `_WAR_DOMAIN_*`: hostilities / armed conflict / refugee flows / war
  economy / sanctions regime / military mobilisation.

**Files:** `puremacro/narrative/indices/_lexicons_expanded.py`,
`puremacro/tests/test_narrative_lexicons_expanded.py`.

**Purpose:** stability-report perturbation only. Not used by any
production index helper. Consumed exclusively by
`NarrativeStabilityReport` (Sprint α-14) to measure how sensitive the
headline cross-country correlations are to lexicon-curation choices.

**Caveat:** non-English additions are best-effort translations of the
English seed, not curated against per-language anchor docs. This is
intentional — the goal is a robustness perturbation, not a production
lexicon. The base `_lexicons.py` (with full reviewer-doc backing) is
not modified.
