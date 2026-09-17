# Macroeconomía Avanzada (MAV) — cuadernos del curso

*English version below.*

Cuadernos de Jupyter del curso de Macroeconomía Avanzada (MAV) de la Maestría en
Economía del ITAM, otoño 2026. Todo el curso corre en Python con
[`puremacro`](https://github.com/jalonso1979/puremacro): cada cuaderno trae sus salidas ya
ejecutadas, así que se puede leer en GitHub sin correr nada.

La carpeta contiene los cuadernos en español (`*_es.ipynb`) y en inglés, los
auxiliares `_datos.py`, `_nbstyle.py` y `_tutor.py`, los datos congelados en
`data_curso/` y los modelos `.mod` en `modelos/`. Las diapositivas en PDF están
en `curso/site/diapositivas/`.

## Abrir un cuaderno en Google Colab

Haga clic en el enlace del cuaderno. La primera celda de código instala
`puremacro` y descarga sólo esta carpeta del repositorio (un clon parcial, unos
55 MB, casi todo datos); no pide acceso a Google Drive. Ejecute las celdas en orden.

| Código | Tema | Español | Inglés |
|---|---|---|---|
| T00 | Syllabus interactivo del curso | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T00_syllabus_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T00_syllabus.ipynb) |
| T01_A | Hechos del ciclo económico desde el mercado de trabajo | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_A_business_cycle_facts_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_A_business_cycle_facts.ipynb) |
| T01_B | Contabilidad nacional y hechos estilizados internacionales | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_B_hechos_estilizados_internacionales_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_B_cross_country_facts.ipynb) |
| T01_C | El ciclo a frecuencia mensual, semanal y diaria | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_C_alta_frecuencia_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_C_high_frequency.ipynb) |
| T02_A | Modelo neoclásico de crecimiento: log-linealización y aproximación local | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_A_neoclassical_growth_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_A_neoclassical_growth_local.ipynb) |
| T02_B | RBC canónico: simulación, momentos y el debate de las horas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_B_rbc_dynare_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_B_rbc_dynare_local.ipynb) |
| T02_C | Precios rígidos y búsqueda: fricciones en bienes y trabajo | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_C_new_keynesian_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_C_new_keynesian_local.ipynb) |
| T02_D | Restricciones ocasionalmente activas y límite inferior cero (OccBin) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_D_occbin_zlb_regimes_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_D_occbin_zlb_regimes.ipynb) |
| T02_E | Estimación bayesiana de modelos DSGE y comparación de modelos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_E_bayesian_dsge_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_E_bayesian_dsge_local.ipynb) |
| T02_F | Fricciones macrofinancieras: Gertler y Karadi (2011) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_F_gertler_karadi_macrofinance_local_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_F_gertler_karadi_macrofinance_local.ipynb) |
| T03_A | Choques agregados con mercados incompletos: Krusell–Smith y jacobianos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_A_aggregate_shocks_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_A_aggregate_shocks.ipynb) |
| T03_B | Volatilidad: GARCH y efecto apalancamiento | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_B_garch_volatility_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_B_garch_volatility.ipynb) |
| T03_C | Crecimiento en riesgo (growth-at-risk) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_C_growth_at_risk_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_C_growth_at_risk.ipynb) |
| T03_D | Incertidumbre de política económica a partir de texto | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_D_narrative_uncertainty_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_D_narrative_uncertainty.ipynb) |
| T03_E | Modelos de lenguaje locales como instrumento de medición | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_E_local_llm_uncertainty_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_E_local_llm_uncertainty.ipynb) |
| T03_F | Comunicación de bancos centrales y tono de la política monetaria | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_F_narrative_llm_central_banks_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_F_narrative_llm_central_banks.ipynb) |
| T04_A | El multiplicador de impuestos, de tres maneras | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_A_tax_multiplier_three_ways_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_A_tax_multiplier_three_ways.ipynb) |
| T04_C | Contabilidad del ciclo económico: las cuñas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_C_contabilidad_ciclo_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_C_business_cycle_accounting.ipynb) |
| T05_A | Identificación de choques con SVAR | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_A_svar_identification_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_A_svar_identification.ipynb) |
| T05_B | Proyecciones locales | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_B_local_projections_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_B_local_projections.ipynb) |
| T05_C | Diferencias en diferencias escalonadas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_C_staggered_did_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_C_staggered_did.ipynb) |
| T05_D | DiD con proyecciones locales (LP-DiD) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_D_lp_did_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_D_lp_did.ipynb) |
| T05_E | Transmisión dependiente del estado y respuestas generalizadas | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_E_regime_girf_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_E_regime_girf.ipynb) |
| T05_F | SVAR de la inflación de 2021–2024 (Bernanke–Blanchard) | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_F_svar_post_covid_inflation_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_F_svar_post_covid_inflation.ipynb) |
| T06_A | ¿De dónde viene la desigualdad de riqueza? | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_A_wealth_inequality_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_A_wealth_inequality.ipynb) |
| T06_B | Ciclo de vida, demografía y pensiones | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_B_life_cycle_and_demographics_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_B_life_cycle_and_demographics.ipynb) |
| T06_C | Carteras, preferencias y la frontera HANK | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_C_portfolios_and_preferences_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_C_portfolios_and_preferences.ipynb) |
| T07_A | Dinámica empresarial y mala asignación de recursos | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T07_A_firm_dynamics_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T07_A_firm_dynamics.ipynb) |
| T08_A | Búsqueda laboral y flujos de trabajadores | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T08_A_labor_flows_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T08_A_labor_flows.ipynb) |
| T09_A | Galería de validación numérica | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_A_validation_gallery_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_A_validation_gallery.ipynb) |
| T09_B | Construye tu propio índice | [español](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_B_build_your_own_index_es.ipynb) | [inglés](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_B_build_your_own_index.ipynb) |

## Correr los cuadernos en su máquina

Requiere Python 3.11 o posterior.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/jalonso1979/puremacro.git
cd puremacro
git sparse-checkout set curso/notebooks
python -m pip install "puremacro[io]>=4.0.1,<5" statsmodels arch jupyterlab
cd curso/notebooks
jupyter lab
```

Un `git clone https://github.com/jalonso1979/puremacro.git` completo también funciona. T03_E usa un modelo de lenguaje local si lo
encuentra (`pip install "puremacro[local-llm]"` u Ollama); sin él corre con un
motor simulado que avisa y no extrae eventos, que es lo que muestran sus salidas
guardadas. Ningún cuaderno necesita red: las descargas en vivo son opcionales y,
por omisión, se leen los datos congelados de `data_curso/`.

## Datos

Los archivos de `data_curso/` son copias congeladas de datos públicos de
terceros (FRED, OCDE, INEGI, ILOSTAT, Banco Mundial y otros). La licencia MIT de
`puremacro` cubre el código, **no** esos datos. La procedencia de cada archivo y
cómo citarlo están en [`data_curso/FUENTES.md`](data_curso/FUENTES.md). Si usa
estos datos, cite a los editores originales.

## Licencia

El código es MIT. El texto de los cuadernos y las diapositivas del curso están bajo
[Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.es):
puede reutilizarlos y adaptarlos, también con fines comerciales, citando
«Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM». Los datos conservan los
términos de sus editores.

---

# Advanced Macroeconomics (MAV) — course notebooks

Jupyter notebooks for the Advanced Macroeconomics (MAV) course in ITAM's Master's
programme in Economics, fall 2026. The whole course runs in Python with
[`puremacro`](https://github.com/jalonso1979/puremacro). Every notebook is saved with its
executed outputs, so you can read it on GitHub without running anything.

This folder holds the notebooks in Spanish (`*_es.ipynb`) and English, the
helpers `_datos.py`, `_nbstyle.py` and `_tutor.py`, the frozen data in
`data_curso/` and the `.mod` models in `modelos/`. The PDF slides are in
`curso/site/diapositivas/` (in Spanish).

## Open a notebook in Google Colab

Click the notebook's link. The first code cell installs `puremacro` and fetches
only this folder from the repository (a partial clone of about 55 MB, mostly
data); it does not ask for Google Drive access. Run the cells in order.

| Code | Topic | Spanish | English |
|---|---|---|---|
| T00 | Interactive course syllabus | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T00_syllabus_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T00_syllabus.ipynb) |
| T01_A | Business-cycle facts from the labour side | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_A_business_cycle_facts_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_A_business_cycle_facts.ipynb) |
| T01_B | National accounts and cross-country stylized facts | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_B_hechos_estilizados_internacionales_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_B_cross_country_facts.ipynb) |
| T01_C | The cycle at monthly, weekly and daily frequency | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_C_alta_frecuencia_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T01_C_high_frequency.ipynb) |
| T02_A | Neoclassical growth model: log-linearization and local approximation | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_A_neoclassical_growth_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_A_neoclassical_growth_local.ipynb) |
| T02_B | Canonical RBC: simulation, moments and the hours debate | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_B_rbc_dynare_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_B_rbc_dynare_local.ipynb) |
| T02_C | Sticky prices and search: frictions in goods and labour markets | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_C_new_keynesian_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_C_new_keynesian_local.ipynb) |
| T02_D | Occasionally binding constraints and the zero lower bound (OccBin) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_D_occbin_zlb_regimes_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_D_occbin_zlb_regimes.ipynb) |
| T02_E | Bayesian DSGE estimation and model comparison | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_E_bayesian_dsge_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_E_bayesian_dsge_local.ipynb) |
| T02_F | Macro-financial frictions: Gertler–Karadi (2011) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_F_gertler_karadi_macrofinance_local_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T02_F_gertler_karadi_macrofinance_local.ipynb) |
| T03_A | Aggregate shocks with incomplete markets: Krusell–Smith and sequence-space Jacobians | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_A_aggregate_shocks_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_A_aggregate_shocks.ipynb) |
| T03_B | Volatility: GARCH and leverage effects | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_B_garch_volatility_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_B_garch_volatility.ipynb) |
| T03_C | Growth-at-risk | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_C_growth_at_risk_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_C_growth_at_risk.ipynb) |
| T03_D | Economic policy uncertainty from text | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_D_narrative_uncertainty_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_D_narrative_uncertainty.ipynb) |
| T03_E | Local language models as measurement tools | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_E_local_llm_uncertainty_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_E_local_llm_uncertainty.ipynb) |
| T03_F | Central-bank communication and monetary-policy tone | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_F_narrative_llm_central_banks_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T03_F_narrative_llm_central_banks.ipynb) |
| T04_A | The tax multiplier, three ways | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_A_tax_multiplier_three_ways_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_A_tax_multiplier_three_ways.ipynb) |
| T04_C | Business-cycle accounting: the wedges | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_C_contabilidad_ciclo_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T04_C_business_cycle_accounting.ipynb) |
| T05_A | Identifying shocks with SVARs | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_A_svar_identification_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_A_svar_identification.ipynb) |
| T05_B | Local projections | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_B_local_projections_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_B_local_projections.ipynb) |
| T05_C | Staggered difference-in-differences | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_C_staggered_did_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_C_staggered_did.ipynb) |
| T05_D | Difference-in-differences with local projections (LP-DiD) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_D_lp_did_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_D_lp_did.ipynb) |
| T05_E | State-dependent transmission and generalized impulse responses | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_E_regime_girf_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_E_regime_girf.ipynb) |
| T05_F | SVAR of the 2021–2024 inflation (Bernanke–Blanchard) | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_F_svar_post_covid_inflation_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T05_F_svar_post_covid_inflation.ipynb) |
| T06_A | Where does wealth inequality come from? | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_A_wealth_inequality_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_A_wealth_inequality.ipynb) |
| T06_B | Life cycle, demographics and pensions | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_B_life_cycle_and_demographics_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_B_life_cycle_and_demographics.ipynb) |
| T06_C | Portfolios, preferences and the HANK frontier | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_C_portfolios_and_preferences_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T06_C_portfolios_and_preferences.ipynb) |
| T07_A | Firm dynamics and misallocation | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T07_A_firm_dynamics_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T07_A_firm_dynamics.ipynb) |
| T08_A | Labour search and worker flows | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T08_A_labor_flows_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T08_A_labor_flows.ipynb) |
| T09_A | Numerical validation gallery | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_A_validation_gallery_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_A_validation_gallery.ipynb) |
| T09_B | Build your own index | [Spanish](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_B_build_your_own_index_es.ipynb) | [English](https://colab.research.google.com/github/jalonso1979/puremacro/blob/main/curso/notebooks/T09_B_build_your_own_index.ipynb) |

## Run the notebooks locally

Requires Python 3.11 or later.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/jalonso1979/puremacro.git
cd puremacro
git sparse-checkout set curso/notebooks
python -m pip install "puremacro[io]>=4.0.1,<5" statsmodels arch jupyterlab
cd curso/notebooks
jupyter lab
```

A full `git clone https://github.com/jalonso1979/puremacro.git` also works. T03_E uses a local language model if it finds one
(`pip install "puremacro[local-llm]"` or Ollama); without one it runs on a mock
engine that prints a notice and extracts no events, which is what its saved outputs
show. No notebook needs the network: live downloads are optional and, by default,
the frozen data in `data_curso/` is read.

## Data

The files in `data_curso/` are frozen copies of public third-party data (FRED,
OECD, INEGI, ILOSTAT, World Bank and others). The MIT licence of `puremacro`
covers the code, **not** that data. Where each file comes from, and how to cite
it, is in [`data_curso/FUENTES.md`](data_curso/FUENTES.md). If you use this data,
cite the original publishers.

## Licence

The code is MIT. The notebook text and the course slides are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
you may reuse and adapt them, commercially too, crediting
"Jorge Alonso-Ortiz, *Macroeconomía Avanzada*, ITAM". The data keep their publishers' terms.
