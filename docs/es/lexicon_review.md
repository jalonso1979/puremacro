> 🇬🇧 [English](../lexicon_review.md) · 🇪🇸 Español

# Registro de Revisión de Léxicos

Este registro documenta quién revisó cada léxico lingüístico, con qué documentos
de referencia y en qué fecha. Es obligatorio para cualquier léxico `_<DOMAIN>_<lang>`
distribuido bajo `puremacro.narrative.indices._lexicons`.

## `_TECH_DOMAIN`

| Idioma | Revisor | Documentos de referencia (≥ 3) | Fecha |
|---|---|---|---|
| en | Jorge Alonso Ortiz | BBD AI-EPU 2023 paper §2; Felten-Raj-Seamans 2021 SMJ Tables 1–2; 3 FOMC speeches Powell (2023-05, 2024-01, 2024-06) referencing AI | 2026-05-11 |
| es | Jorge Alonso Ortiz | Banxico Informe Trimestral 2024-Q1 §III; BCRA Informe de Política Monetaria 2024-03; CEPAL "AI en el mercado laboral latinoamericano" 2023 | 2026-05-11 |
| pt | Jorge Alonso Ortiz | BCB COPOM Ata 2024-09 (minutes 264); BCB Relatório de Inflação 2024-Q2 §3; Discurso Roberto Campos Neto 2024-04 (digitalização) | 2026-05-11 |
| de | Jorge Alonso Ortiz | Bundesbank Monatsbericht 2024-04 (KI-Wirtschaft); ECB Lagarde speech 2023-10-25 (AI / labour); Bundesbank Speech Mauderer 2024-03 | 2026-05-11 |
| fr | Jorge Alonso Ortiz | Banque de France Bulletin 2024-Q1 §2 (numérique); ECB Lagarde 2024 Sintra opening remarks; Trésor Trésor-Éco n°337 (mai 2024) | 2026-05-11 |
| it | Jorge Alonso Ortiz | Banca d'Italia Relazione Annuale 2023 §III (digitalizzazione); Banca d'Italia speech Visco 2024-02; ISTAT Rapporto Annuale 2024 §3 | 2026-05-11 |
| ja | Jorge Alonso Ortiz | BoJ Outlook Report 2024-10 (生成AI section); BoJ speech Ueda 2024-05; Cabinet Office White Paper on the Economy 2024 | 2026-05-11 |
| zh | Jorge Alonso Ortiz | PBoC 2024Q1 Monetary Policy Report §专栏; PBoC Pan Gongsheng 2024 speech (人工智能); China Statistical Yearbook 2024 ch.20 | 2026-05-11 |

## 2026-05-12 — variantes de estabilidad α.2

**Revisor:** Jorge Alonso Ortiz
**Alcance:** variantes de léxico `_*_EXPANDED` para los 5 índices en scope
(`lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`) × 8 idiomas. Se modificaron seis
familias — `_LABOR_DOMAIN`, `_UNCERTAINTY_TONE`, `_TECH_DOMAIN`,
`_TECH_LABOR_UPSIDE`, `_TECH_LABOR_DOWNSIDE`, `_WAR_DOMAIN` — para un
total de 48 frozensets `_EXPANDED` distintos (más 8 passthroughs `_LUI_PHRASES_*`
reutilizados sin cambios).

Cada expansión añade ≤ `floor(0.20 × |base|)` términos (límite aplicado por
`puremacro/tests/test_narrative_lexicons_expanded.py`), obtenidos a partir de una
lista semilla en inglés por familia y traducidos a cada idioma objetivo:

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

**Archivos:** `puremacro/narrative/indices/_lexicons_expanded.py`,
`puremacro/tests/test_narrative_lexicons_expanded.py`.

**Propósito:** perturbación para informes de estabilidad únicamente. Ningún índice
en producción los utiliza. Son consumidos exclusivamente por
`NarrativeStabilityReport` (Sprint α-14) para medir la sensibilidad de las
correlaciones internacionales de encabezado ante las elecciones de curación del
léxico.

**Advertencia:** las adiciones en idiomas distintos del inglés son traducciones de
la semilla inglesa realizadas con el mejor esfuerzo disponible, sin respaldo en
documentos de referencia por idioma. Esto es intencional: el objetivo es una
perturbación de robustez, no un léxico de producción. El archivo base `_lexicons.py`
(con respaldo completo de documentos revisados) no se modifica.
