> 🇪🇸 Español · 🇬🇧 [English](../notebooks.md)

# Cuadernos de Demostración

`puremacro` incluye cuadernos de demostración interactivos con calidad de publicación que cubren los principales paradigmas de macroeconomía con agentes heterogéneos, econometría empírica, estimación bayesiana, aplicaciones especializadas de clima e historia macroeconómica y suites de frontera para política macroeconómica aplicada.

Cada cuaderno está escrito en formato Jupytext percent (`.py`), se ejecuta de forma determinista con semillas fijas, cumple con el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`) y cuenta con una edición gemela en español (`_es.py`).

---

## La Arquitectura Pedagógica de 7 Secciones

Siguiendo `notebooks/_TEMPLATE.md`, los cuadernos profundizados y de frontera siguen un flujo estructural uniforme de 7 celdas:

1. **Pregunta Motivadora**: 1–2 oraciones que definen el problema económico.
2. **El Método en Matemáticas**: Ecuaciones estructurales y econométricas en LaTeX riguroso y compacto ($...$ / $$...$$).
3. **Intuición**: Sección explícita `**Intuición.**` que traduce las ecuaciones algebraicas a mecanismos económicos intuitivos y lógica de identificación.
4. **Código Desarrollado**: Bloques ejecutables en NumPy puro, autocontenidos y comentados explicando el *porqué* de cada decisión.
5. **Lectura de Resultados**: Análisis narrativo en markdown que interpreta directamente los valores numéricos principales, estimaciones y figuras.
6. **Tu Turno**: Ejercicio exploratorio interactivo con controles `# ← modifica esto`, valores por defecto funcionales con aserciones y retos graduados.
7. **¿Qué tan Exhaustivo es Esto?**: Referencias contextuales que vinculan la demostración con otros módulos de `puremacro` y la literatura.

---

## Cuadernos de Frontera en Política Macroeconómica Aplicada (puremacro 3.2)

Las demostraciones `47` a `50` conectan la macroeconometría teórica con la práctica de política en bancos centrales, ministerios de hacienda y mercados financieros.

### `47_applied_central_bank_policy_suite_es`
- **Fuente**: `notebooks/47_applied_central_bank_policy_suite_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Evaluación en tiempo real de la postura monetaria frente a reglas de Taylor (1993, 1999) y especificaciones con inercia.
  - Simulación de políticas contrafactuales: trayectorias macroeconómicas bajo sendas alternativas de tasas de interés.
  - Bandas de proyección en gráficos de abanico (fan charts) incorporando incertidumbre de choques y dispersión posterior de parámetros.
  - Extracción de tono y sentimiento en comunicados de bancos centrales (FOMC, BCE) mediante calificación léxica.

### `48_applied_realtime_nowcasting_and_news_es`
- **Fuente**: `notebooks/48_applied_realtime_nowcasting_and_news_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Nowcasting del PIB trimestral con modelos de factores dinámicos (DFM) siguiendo a Giannone, Reichlin y Small (2008).
  - Ingesta de series en tiempo real con extremos irregulares (ragged edges) combinando frecuencias mensuales y trimestrales.
  - **Descomposición de noticias**: atribución de las revisiones del pronóstico a sorpresas en publicaciones de datos (empleo, producción, encuestas).
  - Pruebas de ortogonalidad de Mankiw-Shapiro (1986) para evaluar si las revisiones representan noticias o ruido.

### `49_applied_macroprudential_gar_and_stress_es`
- **Fuente**: `notebooks/49_applied_macroprudential_gar_and_stress_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Regresión cuantil condicional para Growth-at-Risk (GaR) siguiendo a Adrian, Boyarchenko y Giannone (2019).
  - Ajuste de densidades paramétricas skew-$t$ de Azzalini para capturar el riesgo asimétrico a la baja en horizontes predictivos.
  - Redes de desbordamiento de riesgo sistémico mediante descomposición generalizada de varianza del error de pronóstico (GFEVD) (Diebold y Yílmaz 2014).
  - Escenarios de estrés para colchones de capital macroprudencial y probabilidades de recesión en gráficos de abanico.

### `50_applied_fiscal_multipliers_and_debt_sustainability_es`
- **Fuente**: `notebooks/50_applied_fiscal_multipliers_and_debt_sustainability_es.py` (Inglés: `.py`)
- **Capacidades Clave**:
  - Estimación trimétodo de multiplicadores fiscales: SVAR de Blanchard-Perotti, proyecciones locales narrativas de Romer-Romer y LP-IV de Mertens-Ravn con estadístico $F$ efectivo de primera etapa.
  - Análisis estocástico de sostenibilidad de la deuda soberana (DSA) con gráficos de abanico ante choques conjuntos de crecimiento, inflación y diferencial de tasas $(r - g)$.
  - Pruebas de estrés de la trayectoria de deuda ante escenarios de riesgo climático físico y de transición.

---

## Demostraciones de Frontera en DSGE Bayesiano y HANK (puremacro 3.0 y 3.1)

### `42_dsge_bayesian_estimation_and_diagnostics_es` (Insignia)
- **Fuente**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics_es.py` (Inglés: `.py`)
- **Modelo**: Economía estadounidense de Smets & Wouters (2007, AER) con 7 variables y 7 choques.
- **Capacidades Clave**: Búsqueda de moda con múltiples algoritmos (`lbfgs`, `csminwel`, `cmaes`), diagnósticos de curvatura `mode_check`, muestreo MCMC con diagnósticos $\hat{R}$ de Gelman-Rubin, suavizador de Kalman para descomposición de choques, pronósticos dinámicos y comparación de modelos por densidad marginal de Laplace y Geweke.

### `43_dsge_nuts_and_analytic_gradients_es`
- **Fuente**: `notebooks/43_dsge_nuts_and_analytic_gradients_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Recursión analítica exacta del gradiente de verosimilitud de Kalman ($\nabla_\theta \ln L$) con solucionadores de Sylvester generalizados, muestreador No-U-Turn (NUTS) con adaptación dual averaging y diagnósticos energéticos BFMI.

### `44_hank_sequence_space_bridge_es`
- **Fuente**: `notebooks/44_hank_sequence_space_bridge_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Puente de agentes heterogéneos en el espacio de secuencias desde archivos `.mod` de Dynare (`hetagent_block`), distribución estacionaria de riqueza $\mathcal{D}^*(a)$, jacobianos Fake-News ($J_{C,r}, J_{C,Y}$) y transiciones no lineales de Broyden.

### `45_dsge_discretion_dsge_var_and_news_shocks_es`
- **Fuente**: `notebooks/45_dsge_discretion_dsge_var_and_news_shocks_es.py` (Inglés: `.py`)
- **Capacidades Clave**: Política óptima discrecional frente a compromiso (descomposición del sesgo de inflación y estabilización), optimización de previas DSGE-VAR($\lambda$) de Del Negro y Schorfheide (2004) y estado-espacio aumentado para choques anticipados de noticias.

### `46_dsge_particle_filtering_and_markov_switching_es`
- **Fuente**: `notebooks/46_dsge_particle_filtering_and_markov_switching_es.py` (Inglés: `.py`)
- **Capacidades Clave**: OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov de Foerster et al. (2016) con GIRFs analíticas cerradas y filtrado de partículas secuencial Monte Carlo con volatilidad estocástica.

---

## Catálogo Completo de Demostraciones

| Cuaderno | Tema y Metodología | Gemelo en Inglés |
|---|---|---|
| `00_whats_new_in_puremacro_3_0_es` | Hito 3.0: Gradientes analíticos de Kalman, NUTS HMC y puente HANK en espacio de secuencias | `00_whats_new_in_puremacro_3_0` |
| `00_whats_new_in_puremacro_2_0_es` | Hito 2.0: API unificada (`lags`, `horizon`, `ci`), objetos de resultado y exportadores | `00_whats_new_in_puremacro_2_0` |
| `01_wealth_inequality_es` | Mercados incompletos de Aiyagari y Huggett, heterogeneidad permanente en $\beta$, Lorenz y Gini | `01_wealth_inequality` |
| `02_aggregate_shocks_es` | Agregación aproximada de Krusell–Smith, dinámica de transición, agente representativo | `02_aggregate_shocks` |
| `03_life_cycle_and_demographics_es` | Consumo y ahorro en ciclo de vida finito, riqueza de cohortes por edad, mortalidad | `03_life_cycle_and_demographics` |
| `04_firm_dynamics_es` | Equilibrio de industria de Hopenhayn con entrada/salida endógena y selección | `04_firm_dynamics` |
| `05_portfolios_and_preferences_es` | Elección de cartera con dos activos, utilidad recursiva de Epstein–Zin, EGM frente a VFI | `05_portfolios_and_preferences` |
| `06_svar_identification_es` | Identificación en VAR estructural: ordenamiento de Cholesky frente a restricciones de signo | `06_svar_identification` |
| `07_local_projections_es` | Proyecciones locales de Jordà con inferencia HAC Newey-West y respuestas estado-dependientes | `07_local_projections` |
| `08_garch_volatility_es` | GARCH(1,1) por máxima verosimilitud y correlación condicional dinámica (DCC) de Engle | `08_garch_volatility` |
| `09_growth_at_risk_es` | Growth-at-Risk de Adrian-Boyarchenko-Giannone mediante regresión cuantil y densidad skew-$t$ | `09_growth_at_risk` |
| `10_staggered_did_es` | Diferencias en diferencias modernas: estimadores de Callaway-Sant'Anna y Sun-Abraham | `10_staggered_did` |
| `11_narrative_uncertainty_es` | Construcción de índices textuales de Incertidumbre de Política Económica (EPU) | `11_narrative_uncertainty` |
| `12_validation_gallery_es` | Cuadro de mando de validación comparando puremacro contra referencias independientes | `12_validation_gallery` |
| `13_build_your_own_index_es` | Construcción multirreceta de índices: Texto $\to$ EPU, Panel $\to$ JLN, Financiero $\to$ FCI | `13_build_your_own_index` |
| `14_tax_multiplier_three_ways_es` | Multiplicador fiscal: SVAR Blanchard-Perotti, LP narrativa Romer-Romer, LP-IV Mertens-Ravn | `14_tax_multiplier_three_ways` |
| `15_lp_did_es` | Proyecciones locales con diferencias en diferencias (LP-DiD) bajo adopción escalonada | `15_lp_did` |
| `16_regime_girf_es` | Respuestas a impulsos generalizadas (GIRF) para VAR de umbral y cambio de régimen de Markov | `16_regime_girf` |
| `17_identification_spec_curve_es` | Curvas de especificación de identificación para análisis de sensibilidad | `17_identification_spec_curve` |
| `18_beveridge_curve_es` | Curva de Beveridge, eficiencia del emparejamiento y desplazamientos vacantes-desempleo | `18_beveridge_curve` |
| `19_model_confidence_set_es` | Conjunto de confianza de modelos (MCS) de Hansen-Lunde-Nason para pronósticos | `19_model_confidence_set` |
| `20_unit_roots_with_power_es` | Pruebas de raíz unitaria de alta potencia DF-GLS de Elliott-Rothenberg-Stock y Ng-Perron | `20_unit_roots_with_power` |
| `21_dynare_vfi_dsl_es` | Especificación declarativa en DSL para VFI con aceleración de Howard | `21_dynare_vfi_dsl` |
| `22_continuous_time_hjb_es` | Esquema en diferencias finitas para Hamilton-Jacobi-Bellman en tiempo continuo | `22_continuous_time_hjb` |
| `23_aiyagari_endogenous_labor_es` | Equilibrio general de mercados incompletos con oferta laboral endógena | `23_aiyagari_endogenous_labor` |
| `24_synthetic_control_es` | Control sintético de Abadie et al. con pruebas de placebo espaciales y temporales | `24_synthetic_control` |
| `25_frequency_connectedness_es` | Redes de desbordamiento en el dominio de la frecuencia de Baruník y Křehlík | `25_frequency_connectedness` |
| `26_cycles_and_bandpass_es` | Filtros de ciclo de Baxter-King, Christiano-Fitzgerald y Beveridge-Nelson | `26_cycles_and_bandpass` |
| `27_garch_midas_macro_risk_es` | Modelado de volatilidad de frecuencias mixtas GARCH-MIDAS de Engle-Ghysels-Sohn | `27_garch_midas_macro_risk` |
| `28_weak_iv_anderson_rubin_es` | Conjuntos de confianza Anderson-Rubin robustos a instrumentos débiles para LP-IV | `28_weak_iv_anderson_rubin` |
| `29_synthetic_did_es` | Diferencias en diferencias sintéticas (SDID) de Arkhangelsky et al. | `29_synthetic_did` |
| `30_narrative_bursts_and_transcripts_es` | Detección de ráfagas de Kleinberg en transcripciones de discursos de bancos centrales | `30_narrative_bursts_and_transcripts` |
| `31_sequence_space_hank_es` | Método jacobiano en el espacio de secuencias de Auclert et al. (2021) para HANK | `31_sequence_space_hank` |
| `32_climate_macro_dice_es` | Modelo DICE de Nordhaus e impuestos óptimos al carbono | `32_climate_macro_dice` |
| `33_gdp_nowcasting_news_es` | Nowcasting del PIB con modelo de factores dinámicos y descomposición de noticias | `33_gdp_nowcasting_news` |
| `34_penalized_macro_forecasting_es` | Elastic Net y Lasso adaptativo para pronósticos macroeconómicos en alta dimensión | `34_penalized_macro_forecasting` |
| `35_empirical_benchmark_replications_es` | Cuadro de mando de replicación en literaturas de referencia de SVAR, LP y DiD | `35_empirical_benchmark_replications` |
| `36_climate_sovereign_debt_risk_es` | Transmisión del riesgo climático físico y de transición a la sostenibilidad de deuda | `36_climate_sovereign_debt_risk` |
| `37_central_bank_narrative_sentiment_es` | Sentimiento en comunicación de bancos centrales y proyecciones locales de alta frecuencia | `37_central_bank_narrative_sentiment` |
| `38_real_time_vintages_and_revisions_es` | Cuentas nacionales en tiempo real en más de 45 países y pruebas noticia vs. ruido | `38_real_time_vintages_and_revisions` |
| `39_multilingual_narrative_harvesting_es` | Recolección narrativa multifuente (50+ conectores) y puntuación macro en 8 idiomas | `39_multilingual_narrative_harvesting` |
| `40_quarterly_national_accounts_es` | Tres enfoques para cuentas nacionales: cambio de año base, identidades y contribuciones | `40_quarterly_national_accounts` |
| `41_dynare_frontier_showcase_es` | Smets-Wouters (2007) con funciones nativas de suavizado y estimación, ZLB y Ramsey | `41_dynare_frontier_showcase` |
| `42_dsge_bayesian_estimation_and_diagnostics_es` | Estimación bayesiana insignia de Smets-Wouters, mode_check, MCMC, fan charts y MDD | `42_dsge_bayesian_estimation_and_diagnostics` |
| `43_dsge_nuts_and_analytic_gradients_es` | Estimación DSGE bayesiana con NUTS y recursión exacta del score de Kalman | `43_dsge_nuts_and_analytic_gradients` |
| `44_hank_sequence_space_bridge_es` | Puente HANK en espacio de secuencias desde archivos `.mod`, jacobianos Fake-News | `44_hank_sequence_space_bridge` |
| `45_dsge_discretion_dsge_var_and_news_shocks_es` | Política discrecional vs compromiso, optimización de previas DSGE-VAR, choques de noticias | `45_dsge_discretion_dsge_var_and_news_shocks` |
| `46_dsge_particle_filtering_and_markov_switching_es` | OccBin diferenciable para NUTS, DSGE con cambio de régimen de Markov y partículas | `46_dsge_particle_filtering_and_markov_switching` |
| `47_applied_central_bank_policy_suite_es` | Postura monetaria, reglas de Taylor contrafactuales, abanicos de proyección, sentimiento | `47_applied_central_bank_policy_suite` |
| `48_applied_realtime_nowcasting_and_news_es` | Nowcasting DFM con series irregulares en tiempo real, descomposición de noticias | `48_applied_realtime_nowcasting_and_news` |
| `49_applied_macroprudential_gar_and_stress_es` | Densidades de Growth-at-Risk (skew-$t$), redes de interconexión financiera y estrés | `49_applied_macroprudential_gar_and_stress` |
| `50_applied_fiscal_multipliers_and_debt_sustainability_es` | Multiplicadores fiscales trimétodo, DSA estocástico soberano bajo estrés climático | `50_applied_fiscal_multipliers_and_debt_sustainability` |

---

## Ejecución y Construcción de Artefactos

Los cuadernos de demostración se pueden ejecutar y compilar en artefactos `.ipynb` prerrenderizados mediante la herramienta de línea de comandos:

```bash
# Ejecutar y compilar todos los cuadernos en .ipynb
python tools/build_notebooks.py

# Compilar un solo cuaderno
python tools/build_notebooks.py 47_applied_central_bank_policy_suite_es

# Ejecutar sin modificar archivos (falla si algún cuaderno genera excepción)
python tools/build_notebooks.py --check
```
