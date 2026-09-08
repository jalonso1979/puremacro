> 🇪🇸 Español · 🇬🇧 [English](../notebooks.md)

# Cuadernos de Demostración

`puremacro` incluye cuadernos de demostración interactivos con calidad de publicación que cubren los principales paradigmas de macroeconomía con agentes heterogéneos, econometría empírica, estimación bayesiana y aplicaciones especializadas de clima e historia macroeconómica.

Cada cuaderno está escrito en formato Jupytext percent (`.py`), se ejecuta de forma determinista con semillas fijas, cumple con el contrato de 4 paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`) y cuenta con una edición gemela en español (`_es.py`).

## Cuaderno Insignia de DSGE Bayesiano

### `42_dsge_bayesian_estimation_and_diagnostics_es`
- **Fuente**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics_es.py`
- **Edición en Inglés**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics.py`
- **Modelo de Referencia**: Economía de EE.UU. de Smets & Wouters (2007, AER 97(3):586–606) con 7 variables observables y 7 choques estructurales.
- **Capacidades Clave Demostradas**:
  1. **Especificación del Modelo e Ingesta de Datos**: Lectura de archivos `.mod` de Dynare (`sw07_pfeifer.mod`) con declaraciones `varobs` y `estimated_params`, e ingesta de 156 trimestres de datos macroeconómicos reales de EE.UU. (`_sw07_data.csv`).
  2. **Búsqueda de la Moda con Múltiples Algoritmos**: Comparación de `"lbfgs"`, `"csminwel"` (búsqueda lineal de Chris Sims con pasos dirigidos por gradiente en entornos no convexos) y `"cmaes"` (Estrategia de Evolución con Adaptación de Matriz de Covarianza) para localizar robustamente las modas posteriores.
  3. **Diagnósticos Visuales de la Moda (`mode_check`)**: Perfiles de corte de curvatura en todas las coordenadas paramétricas para garantizar la concavidad local genuina y diagnosticar problemas de identificación débil.
  4. **Estimación Bayesiana MCMC**: Muestreo Metropolis-Hastings de paseo aleatorio con adaptación de escala, diagnósticos de convergencia $\hat{R}$ de Gelman-Rubin y gráficos de distribución previa frente a posterior.
  5. **Suavizador de Kalman y Descomposición de Choques Estructurales**: Extracción de estados no observados y perturbaciones estructurales suavizadas, verificando la identidad contable histórica de choques sobre el crecimiento del PIB, la inflación y la tasa de política monetaria.
  6. **Pronósticos y Gráficos de Abanico (Fan Charts)**: Pronósticos dinámicos condicionales y fuera de muestra con gráficos de abanico con intervalos de confianza del 90 %.
  7. **Densidad Marginal de los Datos y Comparación de Modelos**: Cálculo de densidades marginales mediante la aproximación asintótica de Laplace y la media armónica modificada de Geweke (1999) en múltiples umbrales de truncamiento, junto con la comparación formal de modelos mediante factores de Bayes.

---

## Cuadernos Modernizados en 2.6.0

### `41_dynare_frontier_showcase_es`
- **Fuente**: `notebooks/41_dynare_frontier_showcase_es.py`
- **Edición en Inglés**: `notebooks/41_dynare_frontier_showcase.py`
- **Mejoras en 2.6.0**: Modernizado para utilizar los métodos nativos de puremacro `.smoother(data)` y `.estimate(data)` sobre `sw07_pfeifer.mod` con datos trimestrales empaquetados de EE.UU., eliminando envoltorios de verosimilitud de juguete. Demuestra descomposiciones de varianza del error de pronóstico (FEVD), regímenes con límite inferior cero (ZLB) lineales por partes con OccBin y transiciones de previsión perfecta de Ramsey.

### `macro_history_and_climate/N12_paleoclimate_eiv_and_simex`
- **Fuente**: `notebooks/macro_history_and_climate/N12_paleoclimate_eiv_and_simex.py`
- **Mejoras en 2.6.0**: Eliminación total de dependencias externas de `statsmodels` en favor de `puremacro.regress.ols` con errores estándar robustos a heterocedasticidad (`HC1`). Implementa extrapolación por simulación (SIMEX) y correcciones por errores en variables (EIV) para reconstrucciones paleoclimáticas de temperatura en NumPy puro.

---

## Catálogo Completo de Demostraciones

| Cuaderno | Tema y Metodología | Gemelo en Inglés |
|---|---|---|
| `01_wealth_inequality_es` | Mercados incompletos de Aiyagari y Huggett, heterogeneidad permanente en $\beta$, curvas de Lorenz e índices de Gini | `01_wealth_inequality` |
| `02_aggregate_shocks_es` | Agregación aproximada de Krusell–Smith, dinámica de transición, agente representativo | `02_aggregate_shocks` |
| `03_life_cycle_and_demographics_es` | Consumo y ahorro en ciclo de vida finito, riqueza de cohortes por edad, ponderación por mortalidad | `03_life_cycle_and_demographics` |
| `04_firm_dynamics_es` | Equilibrio de industria de Hopenhayn con entrada/salida endógena y selección | `04_firm_dynamics` |
| `05_portfolios_and_preferences_es` | Elección de cartera con dos activos, utilidad recursiva de Epstein–Zin, EGM frente a VFI | `05_portfolios_and_preferences` |
| `06_svar_identification_es` | Identificación en VAR estructural: ordenamiento de Cholesky frente a restricciones de signo | `06_svar_identification` |
| `07_local_projections_es` | Proyecciones locales de Jordà con inferencia HAC Newey-West y funciones de respuesta estado-dependientes | `07_local_projections` |
| `08_garch_volatility_es` | GARCH(1,1) por máxima verosimilitud y correlación condicional dinámica (DCC) de Engle en NumPy puro | `08_garch_volatility` |
| `09_growth_at_risk_es` | Growth-at-Risk de Adrian-Boyarchenko-Giannone mediante autorregresión cuantil y densidad skew-$t$ | `09_growth_at_risk` |
| `10_staggered_did_es` | Diferencias en diferencias modernas: estimadores de Callaway-Sant'Anna y Sun-Abraham | `10_staggered_did` |
| `11_narrative_uncertainty_es` | Construcción de índices textuales de Incertidumbre de Política Económica (EPU) en NumPy puro | `11_narrative_uncertainty` |
| `12_validation_gallery_es` | Cuadro de mando de validación comparando puremacro contra referencias independientes | `12_validation_gallery` |
| `13_build_your_own_index_es` | Construcción de incertidumbre multirreceta: Texto $\to$ EPU, Panel $\to$ JLN, Financiero $\to$ FCI | `13_build_your_own_index` |
| `14_tax_multiplier_three_ways_es` | Multiplicador fiscal de EE.UU.: SVAR de Blanchard-Perotti, LP narrativa de Romer-Romer, LP-IV de Mertens-Ravn | `14_tax_multiplier_three_ways` |
| `15_lp_did_es` | Proyecciones locales con diferencias en diferencias (LP-DiD) bajo adopción escalonada | `15_lp_did` |
| `16_regime_girf_es` | Respuestas a impulsos generalizadas (GIRF) para VAR de umbral y cambio de régimen de Markov | `16_regime_girf` |
| `17_identification_spec_curve_es` | Curvas de especificación de identificación para análisis de sensibilidad | `17_identification_spec_curve` |
| `18_beveridge_curve_es` | Curva de Beveridge, eficiencia del emparejamiento y desplazamientos vacantes-desempleo | `18_beveridge_curve` |
| `19_model_confidence_set_es` | Conjunto de confianza de modelos (MCS) de Hansen-Lunde-Nason para pronósticos competitivos | `19_model_confidence_set` |
| `20_unit_roots_with_power_es` | Pruebas de raíz unitaria de alta potencia DF-GLS de Elliott-Rothenberg-Stock y Ng-Perron | `20_unit_roots_with_power` |
| `21_dynare_vfi_dsl_es` | Especificación declarativa en DSL para VFI con aceleración de iteración de políticas de Howard | `21_dynare_vfi_dsl` |
| `22_continuous_time_hjb_es` | Esquema en diferencias finitas para Hamilton-Jacobi-Bellman en tiempo continuo de Achdou et al. | `22_continuous_time_hjb` |
| `23_aiyagari_endogenous_labor_es` | Equilibrio general de mercados incompletos con oferta laboral endógena | `23_aiyagari_endogenous_labor` |
| `24_synthetic_control_es` | Método de control sintético de Abadie et al. con pruebas de placebo espaciales y temporales | `24_synthetic_control` |
| `25_frequency_connectedness_es` | Redes de desbordamiento en el dominio de la frecuencia de Baruník y Křehlík | `25_frequency_connectedness` |
| `26_cycles_and_bandpass_es` | Filtros de ciclo de Baxter-King, Christiano-Fitzgerald y Beveridge-Nelson | `26_cycles_and_bandpass` |
| `27_garch_midas_macro_risk_es` | Modelado de volatilidad de frecuencias mixtas GARCH-MIDAS de Engle-Ghysels-Sohn | `27_garch_midas_macro_risk` |
| `28_weak_iv_anderson_rubin_es` | Conjuntos de confianza Anderson-Rubin robustos a instrumentos débiles para LP-IV | `28_weak_iv_anderson_rubin` |
| `29_synthetic_did_es` | Diferencias en diferencias sintéticas (SDID) de Arkhangelsky et al. | `29_synthetic_did` |
| `30_narrative_bursts_and_transcripts_es` | Detección de ráfagas de Kleinberg en transcripciones de discursos de bancos centrales | `30_narrative_bursts_and_transcripts` |
| `31_sequence_space_hank_es` | Método jacobiano en el espacio de secuencias de Auclert et al. (2021) para modelos HANK | `31_sequence_space_hank` |
| `32_climate_macro_dice_es` | Modelo de evaluación integrada DICE de Nordhaus e impuestos óptimos al carbono | `32_climate_macro_dice` |
| `33_gdp_nowcasting_news_es` | Nowcasting del PIB con modelo de factores dinámicos y descomposición de noticias | `33_gdp_nowcasting_news` |
| `34_penalized_macro_forecasting_es` | Elastic Net y Lasso adaptativo para pronósticos macroeconómicos en alta dimensión | `34_penalized_macro_forecasting` |
| `35_empirical_benchmark_replications_es` | Cuadro de mando de replicación en literaturas de referencia de SVAR, LP y DiD | `35_empirical_benchmark_replications` |
| `36_climate_sovereign_debt_risk_es` | Transmisión del riesgo climático físico y de transición a la sostenibilidad de la deuda | `36_climate_sovereign_debt_risk` |
| `37_central_bank_narrative_sentiment_es` | Calificación de sentimiento en bancos centrales y proyecciones locales de alta frecuencia | `37_central_bank_narrative_sentiment` |
| `38_real_time_vintages_and_revisions_es` | Series en tiempo real de cuentas nacionales en más de 45 países y pruebas noticia vs. ruido | `38_real_time_vintages_and_revisions` |
| `39_multilingual_narrative_harvesting_es` | Recolección narrativa multifuente (más de 50 conectores) y puntuación macro en 8 idiomas | `39_multilingual_narrative_harvesting` |
| `40_quarterly_national_accounts_es` | Tres enfoques para cuentas nacionales: cambio de año base, identidades y contribuciones | `40_quarterly_national_accounts` |
| `41_dynare_frontier_showcase_es` | Smets-Wouters (2007) con funciones nativas 2.6.0 de suavizado y estimación, ZLB y Ramsey | `41_dynare_frontier_showcase` |
| `42_dsge_bayesian_estimation_and_diagnostics_es` | Estimación bayesiana insignia de Smets-Wouters, mode_check, MCMC, fan charts y MDD | `42_dsge_bayesian_estimation_and_diagnostics` |
| `00_whats_new_in_puremacro_2_0_es` | Visión general de la API unificada de puremacro, clases de resultados y exportadores | `00_whats_new_in_puremacro_2_0` |

---

## Ejecución y Construcción de Artefactos

Los cuadernos de demostración se pueden ejecutar y compilar en artefactos `.ipynb` prerrenderizados mediante la herramienta de línea de comandos:

```bash
# Ejecutar y compilar todos los cuadernos en .ipynb
python tools/build_notebooks.py

# Compilar un solo cuaderno
python tools/build_notebooks.py 42_dsge_bayesian_estimation_and_diagnostics_es

# Ejecutar sin modificar archivos (falla si algún cuaderno genera excepción)
python tools/build_notebooks.py --check
```
