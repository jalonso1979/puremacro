> 🇬🇧 [English](../sw07_finite_sample.md) · 🇪🇸 Español

# Diagnósticos de muestra finita para SW07

Este experimento estudia cómo se comporta la [aplicación empírica SW07](empirical_research.md)
con muestras de 156 trimestres generadas por calibraciones explícitas del modelo.
Compara el criterio de distancia mínima observado con criterios simulados,
examina el efecto de estimar las medias y contrasta la covarianza HAC estimada
con la covarianza de muestreo obtenida por Monte Carlo.

La aplicación guarda puntos de recuperación atómicos en el directorio de salida
tras la primera réplica, cada diez réplicas y al completar cada escenario.
Repetir el mismo comando retoma las réplicas guardadas. Si cambian opciones,
código numérico o versiones de dependencias, use `--no-resume` para comenzar
de nuevo. `start_diagnostics_json` en `draws.csv` conserva mensajes y motivos
de diagnóstico de los cuatro puntos iniciales.

El cálculo gaussiano exacto puede usarse sin ejecutar el experimento Monte Carlo:

```python
from puremacro.structural import sw07_finite_sample_moments, simulate_sw07_sample

exact = sw07_finite_sample_moments({}, nobs=156, common_max_lag=4)
print(exact.labels[0], exact.values[0])
print(exact.covariance.shape)  # Covarianza conjunta de 15 momentos estimados.
sample = simulate_sw07_sample({}, nobs=156, seed=20261001)
print(sample.shape)
```

Un diccionario vacío usa la calibración Pfeifer declarada. Los momentos exactos
de primer y segundo orden no hacen normales a estos estimadores cuadráticos.

```bash
python -m puremacro.examples.sw07_finite_sample \
  --replications 399 --seed 20261001 \
  --output research_output/sw07_finite_sample
```

El experimento requiere más tiempo que cargar una referencia numérica:
cada réplica vuelve a estimar dos parámetros desde cuatro puntos iniciales.
Un número pequeño de réplicas sirve para revisar el flujo de trabajo; para
interpretar frecuencias de cola, debe usarse un número acorde con la precisión
Monte Carlo deseada. El comando informa el progreso y no descarga datos.

## Diseño que se conserva

Cada réplica mantiene la longitud de muestra, las medias estimadas usando
toda la muestra, la ventana común de productos, los nueve momentos ajustados,
los seis momentos reservados, los límites de parámetros y la configuración
del optimizador. La covarianza de ajuste predeterminada usa HAC de Bartlett
con ancho de banda ocho. Las innovaciones del modelo son gaussianas e
independientes y el estado inicial se extrae de la distribución estacionaria.
Se excluye el error económico de medición, como en la aplicación empírica.

Los parámetros estimados son `crr`, suavización de la tasa de interés, y
`em`, desviación estándar de la innovación monetaria. Los demás coeficientes
y varianzas de innovaciones permanecen fijos y se registran explícitamente.
Las tres series usan unidades porcentuales trimestrales.

| Escenario | Parámetros generadores | Interpretación |
|---|---|---|
| `baseline_fixed` | Calibración de Pfeifer declarada | Diagnóstico de este modelo fijo |
| `published_fixed` | Moda publicada de SW07, redondeada | Diagnóstico de este modelo fijo |
| `baseline_fitted` | Calibración base con `crr` y `em` estimados en los datos observados | Diagnóstico descriptivo con parámetros sustituidos |
| `published_fitted` | Calibración publicada con `crr` y `em` estimados en los datos observados | Diagnóstico descriptivo con parámetros sustituidos |

Para elegir varios escenarios, repita la opción:

```bash
python -m puremacro.examples.sw07_finite_sample \
  --scenario published_fixed --scenario published_fitted \
  --replications 399 --seed 20261001 --bandwidth 8 \
  --output research_output/sw07_published_diagnostics
```

Los escenarios ajustados dependen de la muestra observada. Sus frecuencias
de excedencia **no son valores p de bootstrap validados ni pruebas de una
hipótesis nula compuesta**. La calibración publicada procede de datos
históricos que se superponen con esta muestra y su incertidumbre no se
propaga. El ejercicio no identifica choques monetarios mediante instrumentos
externos ni reproduce la distribución posterior bayesiana del artículo.

## Interpretación de los resultados

El reporte conserva los ajustes observados, las estimaciones simuladas,
los diagnósticos del optimizador, los límites activos y los errores de cálculo.
Convergencia numérica y adecuación empírica son resultados distintos.
Un rechazo científico es un resultado válido; el comando devuelve un estado
distinto de cero por errores de cálculo, no por resultados desfavorables al modelo.

Las frecuencias de cola se acompañan de **intervalos binomiales Monte Carlo**.
Estos intervalos describen la variación causada por un número finito de
simulaciones. No son intervalos de confianza para `crr` o `em` y excluyen
incertidumbre de calibración, versión de datos, especificación y estabilidad
de régimen. Cero excedencias simuladas no significa probabilidad poblacional
de cola igual a cero. Una réplica fallida no puede contarse silenciosamente
como una no excedencia exitosa.

Los datos observados son la versión revisada de FRED incluida en el paquete,
1966T1–2004T4. La estacionariedad y los parámetros constantes durante ese
periodo son supuestos. Las muestras simuladas los cumplen por construcción;
las observaciones históricas no necesariamente.

## Diagnóstico completado: 2026-10-01

Los cuatro escenarios usaron 399 réplicas cada uno y la semilla 20261001.
Se construyeron los momentos y las matrices HAC de las 1,596 muestras; 22
ajustes quedaron sin resolver numéricamente. Los rangos cuentan cada ajuste
sin resolver primero como no excedencia y luego como excedencia. Los intervalos
Monte Carlo incorporan además la variación aleatoria de la simulación.

| Escenario | Criterio observado | Sin resolver / 399 | Rango de frecuencia de excedencia | Intervalo Monte Carlo conservador al 95% |
|---|---:|---:|---:|---:|
| `baseline_fixed` | 396,736.10 | 4 | 0–1.00% | 0–2.55% |
| `published_fixed` | 25.26 | 9 | 41.60–43.86% | 36.72–48.88% |
| `baseline_fitted` | 396,736.10 | 2 | 0–0.50% | 0–1.80% |
| `published_fitted` | 25.26 | 7 | 39.35–41.10% | 34.52–46.11% |

La calibración base declarada conserva una discrepancia extrema. Con la
calibración publicada fija, el criterio observado es común en la distribución
simulada, pese al pequeño valor p asintótico de J del estudio previo. La regla
ji cuadrada nominal al 5% rechaza entre 62.16% y 64.41% de las muestras generadas
por este modelo. Incluso entre los 383 ajustes numéricamente regulares,
rechaza 241 (62.92%). La regularidad numérica no corrige esta distorsión en
muestra finita.

Con esa calibración publicada, el promedio HAC con ancho de banda ocho
representa solo 55.6% y 52.5% de las varianzas de muestreo exactas de los
estimadores de varianza de inflación y de la tasa de interés. Estimar las
medias también reduce sus esperanzas en 0.67 y 0.70 desviaciones estándar de
muestreo. Son diagnósticos que contribuyen a la interpretación, no una
descomposición causal de la tasa de rechazo. Aumentar el ancho de banda HAC
por sí solo no se ha validado como solución.

Este resultado cambia la interpretación del diagnóstico asintótico previo;
no establece la adecuación del modelo ni restablece intervalos de confianza
para los parámetros. El expediente del repositorio
`reviews/2026-10-01-sw07-finite-sample/` conserva el protocolo, las réplicas,
las referencias exactas, la auditoría de fuentes, los hashes y los reportes.

El [experimento emparejado posterior](sw07_estimator_experiment.md) separa
correcciones de esperanzas de muestra finita y ponderaciones de covarianza.
La validación independiente encuentra mejora importante con covarianzas del
proceso generador conocido, pero ni estas restablecen la referencia ji cuadrada
nominal.

## Momentos y ponderación

`moment_audit.csv` distingue los momentos estacionarios, sus esperanzas exactas
en muestra finita y los momentos empíricos simulados. Las diferencias reflejan la construcción en muestra
finita, incluida la estimación de medias, sin modificar la solución de Lyapunov.
La matriz de observación transforma niveles del PIB en crecimiento y conserva
inflación y tasa de interés trimestrales en niveles, conforme a las
[ecuaciones de medición y el apéndice de datos de SW07](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf).

`covariance_audit.csv` compara la covarianza gaussiana exacta en muestra finita,
la covarianza Monte Carlo de los estimadores de momentos y el promedio de sus
covarianzas HAC estimadas. Se revisan
anchos de banda cuatro, ocho y doce sin volver a estimar parámetros para cada
uno. Se conserva la dependencia entre momentos. Estas matrices son
covarianzas de estimadores, no covarianzas de largo plazo sin escalar.

La calibración declarada difiere de la moda publicada. Por ejemplo, el
[código de Pfeifer](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod)
fija la desviación estándar del choque de prima de riesgo en 1.8513 dentro
del bloque de choques, valor distinto del punto inicial de estimación y de la
moda del artículo. La auditoría de datos y medición encontró que esta escala
domina la varianza del crecimiento del PIB en la calibración declarada.
La persistencia tecnológica y las propiedades de los momentos centrados
requieren diagnósticos separados.

## Evidencia exportada

| Archivo | Contenido |
|---|---|
| `summary.csv` | Resultados por escenario, excedencias, incertidumbre Monte Carlo y fallos |
| `observed_fits.csv` | Ajustes sobre los momentos observados |
| `draws.csv` | Estimaciones, criterio y estado de cada réplica |
| `moment_audit.csv` | Momentos observados, poblacionales y empíricos simulados |
| `covariance_audit.csv` | Covarianza simulada frente al promedio HAC |
| `draw_evidence.npz` | Arreglos completos de momentos y covarianzas por réplica |
| `criterion_distribution.png` | Distribuciones del criterio y valor observado |
| `manifest.json` | Fuentes, semillas, configuración, parámetros, entorno y hashes |
| `report.md` | Reporte generado a partir de los resultados de esa ejecución |

La columna `array_row` de `draws.csv` vincula cada réplica con su fila en los
arreglos NPZ. El manifiesto conserva etiquetas de momentos y dimensiones.
La auditoría de momentos y HAC puede usar momentos construidos correctamente
aunque falle la optimización posterior; sus conteos difieren de los ajustes utilizables.

El reporte incluye sesgo y RMSE entre ajustes utilizables y dos diagnósticos
de reglas asintóticas. La regla ingenua ji cuadrada con siete grados de libertad
incluye ajustes en los límites deliberadamente; no valida inferencia en esos
límites. La frecuencia de rechazo J regular se condiciona al subconjunto
numéricamente regular, cuyo denominador se informa. Ninguno establece tamaño
de prueba o cobertura generales.

JSON representa los diagnósticos numéricos no disponibles mediante `null`.
Las filas de réplicas y los arreglos binarios conservan la evidencia para
auditar resultados ausentes o fallidos. Las pruebas numéricas de referencia
y el experimento estadístico siguen siendo comprobaciones diferentes:
un diagnóstico desfavorable del modelo no demuestra un defecto de software.
