> 🇬🇧 [English](../sw07_estimator_experiment.md) · 🇪🇸 Español

# Controles del estimador SW07

El [diagnóstico de muestra finita](sw07_finite_sample.md) encontró distorsión
grave de la referencia ji cuadrada en muestras generadas por la calibración
publicada de SW07. Este experimento separa cambios en los momentos esperados
del modelo de cambios en sus ponderaciones, usando la misma muestra para
todas las variantes.

| Variante | Momentos del modelo durante el ajuste | Covarianza de ponderación |
|---|---|---|
| `population_hac` | Covarianzas poblacionales estacionarias | HAC8 estimada en cada muestra |
| `finite_hac` | Esperanzas exactas de los momentos centrados muestrales | HAC8 estimada en cada muestra |
| `population_oracle` | Covarianzas poblacionales estacionarias | Covarianza exacta en el proceso generador conocido, fija |
| `finite_oracle` | Esperanzas exactas de los momentos centrados muestrales | Covarianza exacta en el proceso generador conocido, fija |

Las ponderaciones oráculo son un control diagnóstico. Usan los parámetros
generadores verdaderos; no constituyen un procedimiento factible de estimación
de covarianzas. Momentos exactos de primer y segundo orden no implican
normalidad ni validan una prueba ji cuadrada. No se sustituye automáticamente
el estimador empírico ni se seleccionan intervalos paramétricos.

## Esperanzas exactas rápidas

La función que calcula solo esperanzas evita construir toda la covarianza
de estimadores durante la optimización. Sumas acumuladas de autocovarianzas
tratan exactamente las medias estimadas y la ventana común de productos:

```python
from puremacro.structural import sw07_finite_sample_expectations

expected = sw07_finite_sample_expectations({}, nobs=156, common_max_lag=4)
print(expected.shape)  # 15 esperanzas de momentos de covarianza muestral
```

Un diccionario vacío selecciona la calibración declarada de Pfeifer. En cambio,
el experimento usa explícitamente la calibración publicada completa y
redondeada, con `crr=0.81`, `em=0.24`. Los demás parámetros permanecen fijos
y no se propaga su incertidumbre.

## Fases independientes

El diseño registrado usa 399 muestras de calibración y 999 muestras nuevas
de validación, con 156 trimestres gaussianos estacionarios cada una. Las cuatro
variantes comparten cada muestra y su matriz HAC de 15 momentos. Se conservan
los nueve momentos ajustados, seis reservados, límites y cuatro puntos iniciales.

```bash
python -m puremacro.examples.sw07_estimator_experiment \
  --phase calibration --replications 399 --seed 20261002 \
  --output research_output/sw07_controls/calibration

python -m puremacro.examples.sw07_estimator_experiment \
  --phase validation --replications 999 --seed 20261002 \
  --calibration research_output/sw07_controls/calibration \
  --output research_output/sw07_controls/validation
```

Los índices de fase generan secuencias aleatorias distintas con la misma semilla
principal. Una ejecución pequeña solo comprueba el flujo de trabajo.
`run_sw07_estimator_experiment` ejecuta una fase;
`compare_sw07_estimator_phases` aplica la referencia de calibración a validación.
El comando autentica los hashes antes de cargar la fase de referencia.

Los puntos de recuperación atómicos se guardan tras la primera réplica, cada
diez réplicas y al terminar. Retomar exige idénticas fuentes numéricas, opciones,
datos y versiones de dependencias. `--no-resume` inicia otra ejecución. Un código
de salida distinto de cero informa resultados numéricos sin resolver, aunque
se hayan completado todas las réplicas y exportaciones solicitadas.

## Interpretación

`summary.csv` informa rechazos de la regla ingenua ji cuadrada(7), fallos,
límites y recuperación paramétrica. Los rangos cuentan cada fallo de ambas
maneras. Sesgo, RMSE y cuantiles se condicionan a ajustes utilizables.
`paired_differences.csv` compara errores cuadrados en réplicas utilizables
comunes, informa su denominador y mide variación Monte Carlo. Criterios con
ponderaciones diferentes tienen escalas distintas.

El umbral de calibración usa el rango `ceil(0.95*(B+1))` y rechazo estricto
por encima del estadístico de orden. Los criterios desconocidos se acotan entre
cero e infinito, produciendo un intervalo para el umbral. Los conteos inferior
y superior de validación también conservan los ajustes sin resolver.
`calibrated_validation.csv` registra ambas ambigüedades e intervalos conservadores
de Clopper–Pearson. Estos intervalos se condicionan a la muestra de calibración
obtenida; no recogen toda la incertidumbre de estimar el umbral. Una referencia
muy pequeña o incompleta puede producir límites infinitos, identificados
explícitamente como no acotados.

La validación usa réplicas nuevas, pero de un solo proceso generador conocido.
No establece una prueba de hipótesis compuesta, robustez a parámetros auxiliares,
adecuación económica ni cobertura general. Los ajustes sobre datos FRED revisados
son descriptivos y la calibración publicada se superpone con la muestra histórica.
Un resultado favorable no restablece la inferencia paramétrica convencional.

## Comparación completada: 2026-10-01

La ejecución predefinida completó 399 muestras de calibración y 999 muestras
independientes de validación: 1,398 muestras compartidas por cuatro variantes,
para 5,592 ajustes. La auditoría independiente reprodujo tasas, límites por
fallos, comparaciones emparejadas y umbrales fijados a partir de los archivos.
Los resultados de validación son:

| Variante | Sin resolver / 999 | Rechazo con regla ji cuadrada nominal al 5% | Rechazo con umbral de simulación separado |
|---|---:|---:|---:|
| `population_hac` | 22 | 61.56–63.76% | 2.20–5.51% |
| `finite_hac` | 8 | 40.94–41.74% | 2.30–6.51% |
| `population_oracle` | 0 | 7.71% | 4.90% |
| `finite_oracle` | 0 | 8.21% | 5.91% |

Corregir solo las esperanzas deja una distorsión grave con HAC. Las ponderaciones
oráculo la reducen considerablemente, pero los intervalos Monte Carlo al 95%
de la regla ingenua, 6.13–9.54% y 6.58–10.09%, siguen por encima de 5%.
Por tanto, medias y covarianzas exactas no restablecen la aproximación ji cuadrada.
Los intervalos para umbrales simulados son 3.65–6.43% y 4.53–7.55% en las
variantes oráculo, condicionados a la muestra de calibración registrada y al
proceso generador conocido.

La corrección de esperanzas mejora el error cuadrático emparejado bajo HAC,
pero lo empeora bajo ponderaciones oráculo para ambos parámetros en validación.
No hay mejora universal del riesgo paramétrico. Los 46 ajustes sin resolver
entre ambas fases fallan el control adicional de primer orden pese al éxito
de terminación reportado por el optimizador; permanecen en los límites.
Todas las dinámicas generadoras son estacionarias. No se adopta automáticamente
un estimador ni se restablecen intervalos paramétricos convencionales. Un método
factible con ponderaciones estimadas y la incertidumbre de calibración auxiliar
necesitan validación separada.

`draws.csv` conserva todos los puntos iniciales, semillas y hashes. `array_row`
vincula las cuatro variantes a momentos y matrices HAC comunes en
`draw_evidence.npz`. `manifest.json` conserva la covarianza oráculo, fuentes,
parámetros, entorno y hashes; `report.md` y `estimator_comparison.png` presentan
resultados. El protocolo y la evidencia completa se conservan en
`reviews/2026-10-01-sw07-estimator-experiment/` dentro del repositorio.
