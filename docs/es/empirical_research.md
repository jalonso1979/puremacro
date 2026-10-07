> 🇬🇧 [English](../empirical_research.md) · 🇪🇸 Español

# Estimación estructural con datos reales y replicación publicada

Estos dos estudios amplían los [flujos de investigación](research_workflows.md)
con datos macroeconómicos observados y una especificación original publicada.
Uno evalúa el ajuste de un modelo estructural condicional a momentos empíricos;
el otro comprueba un resultado concreto de un artículo.

## Smets–Wouters: de momentos observados a parámetros

```bash
python -m puremacro.examples.empirical_sw07_matching --output research_output/empirical_sw07
```

Es una nueva aplicación de mínima distancia del modelo Smets–Wouters lineal,
no una replicación del posterior bayesiano del artículo. Los datos son la
**versión revisada de FRED** incluida en el paquete, 1966T1–2004T4. Siguen las
definiciones del [apéndice original](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf),
pero no corresponden a la versión histórica de los autores. El manifiesto
conserva checksum, fecha de construcción, series fuente y transformaciones.
La carga verifica el SHA256 revisado —normalizando los finales de línea—,
las siete columnas fuente y la muestra completa de 156 trimestres. Una versión
distinta requiere una revisión deliberada.

Se estiman la persistencia de la regla monetaria `crr` y la desviación estándar
de la innovación monetaria `em`. Los demás coeficientes y desviaciones estándar
se fijan en la **calibración del modelo de Pfeifer**, registrada explícitamente;
esta no es la moda posterior publicada de SW07. No se estima la
incertidumbre de esa calibración. Los choques son latentes, restringidos por
el modelo; ningún instrumento externo se trata como choque estructural observado.

| Serie | Transformación | Unidad |
|---|---|---|
| Crecimiento del PIB real | `100 * dlog(GDPC1 / CNP16OV)` | Porcentaje trimestral |
| Inflación del deflactor | `100 * dlog(GDPDEF)` | Porcentaje trimestral |
| Tasa federal de fondos | Promedio trimestral de `FEDFUNDS / 4` | Tasa porcentual trimestral |

```python
from puremacro.structural import fit_empirical_sw07

study = fit_empirical_sw07(profile_points=5)
print(study.metadata["status"])
print(study.fit.summary())
print(study.fit.moment_fit(held_out=True))
```

Se ajustan seis covarianzas contemporáneas únicas y tres autocovarianzas de
orden uno. Se reservan las autocovarianzas de órdenes dos y cuatro para
comparaciones descriptivas. Todos los productos usan una ventana común de
152 trimestres, tras reservar cuatro trimestres iniciales, y medias estimadas
con la muestra completa. La covarianza HAC Bartlett conjunta de 15 por 15
conserva la incertidumbre entre momentos:
es la covarianza de sus estimadores, no una covarianza de largo plazo sin
escalar. Se requiere estacionariedad, dependencia débil, momentos de cuarto
orden finitos y un ancho de banda adecuado. La derivada poblacional de primer
orden respecto de la media es cero para covarianzas centrales estacionarias;
este argumento es asintótico, no una corrección exacta de muestra finita.

Con transición `T`, carga de choques `R`, covarianza de innovaciones `Q` y
matriz de observación `Z`, las covarianzas teóricas se obtienen de

\[
P=TPT'+RQR',\qquad \Gamma_h=ZT^hPZ'.
\]

Se excluye la pequeña regularización numérica de medición del filtro de Kalman.
Se verifican estacionariedad y residuo de Lyapunov. Varios valores iniciales y
perfiles del criterio permiten detectar problemas numéricos o de identificación;
no prueban identificación global ni fortaleza estadística.

### Interpretar un ajuste desfavorable

Éxito del optimizador y adecuación empírica son resultados distintos. Los
límites vinculantes, la falta de rango y las fallas numéricas se conservan.
La prueba condicional de sobreidentificación puede rechazar el conjunto de
modelo, calibración y momentos. Ese rechazo es un resultado válido; no se
deben cambiar momentos o tolerancias para forzar aceptación.

La incertidumbre paramétrica convencional se omite por defecto.
`--allow-conditional-inference` la solicita explícitamente, pero el rechazo,
los límites vinculantes u otros diagnósticos fallidos siguen impidiendo
reportarla. No rechazar no demuestra que el modelo sea correcto; los errores estándar disponibles
siguen condicionados a suavidad, identificación, covarianza consistente y
calibración fija. Los perfiles son descriptivos, no intervalos robustos a
identificación débil. Los momentos reservados usan la misma muestra y no son
datos de validación independientes. Cambios estructurales e incertidumbre de
calibración requieren diseños adicionales.

La ejecución por defecto conserva la base y reporta estos resultados:

| Calibración fija | `crr` | `em` | Criterio de mínima distancia | Interpretación |
|---|---:|---:|---:|---|
| Base declarada de Pfeifer | 0.98 | 0.01 | 396736.10 | Ambos límites vinculan; `nonregular_fit`, sin referencia J regular ni errores estándar paramétricos |
| Moda publicada de SW07, exploratoria | 0.83826 | 0.25409 | 25.25541 | Ajuste interior; la referencia J asintótica da `p ≈ 0.000684` (7 grados de libertad), pero la simulación posterior encuentra distorsión grave en muestra finita |

La sensibilidad exploratoria se añadió **después** de observar los límites
vinculantes de la base. Fija los demás parámetros en las columnas completas de
moda posterior de los cuadros 1a/b de SW07 y reestima los mismos dos parámetros,
sin cambiar la base ni los momentos. La reducción del criterio muestra la
dependencia de esta aplicación respecto de la calibración fija; no demuestra
adecuación empírica. Los coeficientes publicados se estimaron con una muestra
histórica superpuesta y no son validación independiente. No se propaga su
incertidumbre ni se reporta inferencia paramétrica normal para esta sensibilidad.

La transición de la base tiene radio espectral 0.9977. Con solo 156 trimestres,
esa persistencia hace importantes las limitaciones de las aproximaciones de
HAC y del centrado en muestra finita, aunque existan momentos poblacionales
estacionarios. Se exportan anchos de banda 4, 8 y 12 y comparaciones descriptivas
de submuestras históricas; no corrigen la inferencia cerca de una raíz unitaria
ni los posibles cambios de régimen. `calibration_sensitivity.csv` compara ambos
ajustes. Las discrepancias estandarizadas dividen por los errores estándar de
los momentos empíricos y son descriptivas, no pruebas t de residuos ajustados.
La ejecución por defecto no reporta intervalos paramétricos normales.

El [diagnóstico posterior de muestra finita](sw07_finite_sample.md) revisa la
correspondencia de datos, el sesgo exacto por estimación de medias, la covarianza
HAC y el mismo ajuste acotado mediante simulaciones gaussianas estacionarias.
Sus frecuencias condicionales no sustituyen los resultados históricos por
p-valores bootstrap validados ni restablecen intervalos paramétricos normales.
En 399 réplicas de la calibración publicada fija, entre 41.60% y 43.86% exceden
el criterio observado, y la regla ji cuadrada nominal al 5% rechaza entre
62.16% y 64.41% de las muestras generadas por el modelo. Estos rangos conservan
los ajustes sin resolver. Por tanto, el pequeño valor p asintótico anterior
no es evidencia fiable de rechazo del modelo con esta longitud de muestra.
La base declarada conserva una discrepancia extrema, dominada por la escala
del choque de prima de riesgo.

## Romer–Romer (2010): respuesta tributaria original

```bash
python -m puremacro.examples.romer_romer_2010_replication --output research_output/rr2010
```

Se comprueba un resultado base de
[Romer y Romer (2010), *The Macroeconomic Effects of Tax Changes*](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763).
El libro original y la especificación `EXOGERNR.RAT` proceden del
[archivo de los autores](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).
Se conservan hashes de las fuentes, los datos derivados y la referencia.

```python
from puremacro.replication import estimate_rr2010_baseline

replication = estimate_rr2010_baseline()
print(replication.to_frame().loc[10])
# Conservar la covarianza completa al usar estas respuestas como momentos.
targets = replication.to_moment_targets()
print(targets.labels)
```

En 1950T1–2007T4 se explica `100 * delta log(PIB real)` con constante y
valores contemporáneos y doce rezagos de `100 * (DEFIC + LONGR) / NOMGDP`.
Las sumas acumuladas de los trece coeficientes tributarios producen la
respuesta del nivel del PIB ante un aumento tributario de uno por ciento
del PIB. Un choque positivo representa un aumento de impuestos.

La especificación original usa covarianza MCO convencional con ajuste por
grados de libertad residuales. La covarianza de respuestas acumuladas aplica
la misma transformación lineal a toda la covarianza de coeficientes. HAC y
proyecciones locales pueden ser extensiones útiles, pero cambian la
especificación publicada.

Se reportan dos comprobaciones distintas:

1. **Paridad numérica:** coeficientes, covarianzas y respuestas acumuladas
   contra una ejecución congelada de software independiente que no importa
   puremacro. La tolerancia solo permite redondeo numérico.
2. **Evidencia publicada:** el mínimo a horizonte diez, respuesta `-3.08` y
   estadístico t `-3.53`, en la página impresa 781. Solo el redondeo publicado
   justifica la tolerancia mayor; no se aplica a la paridad de software.

Los datos originales producen una respuesta a horizonte diez de
**-3.0808127631** y un estadístico t de **-3.5249960750**. La paridad de software
pasa, y coinciden la respuesta y el horizonte mínimo publicados. El estadístico
t queda fuera del intervalo de redondeo de -3.53 por **0.000003925**. La lectura
independiente de los bancos RATS originales corrobora este valor. La causa no
está resuelta; no se redondean valores intermedios ni amplían tolerancias para
forzar coincidencia. El caso publicado y la aplicación informan **FAIL**, con
código de salida distinto de cero. No se afirma coincidencia de todos los objetivos.

El alcance es esta especificación base con datos originales, no todas las
pruebas de robustez ni una validación independiente del supuesto de exclusión
narrativa. El caso anterior `regression.romer_romer_tax_multiplier_ols` es una
LP modificada a horizonte ocho, con otra muestra y datos revisados, y su
comparación gruesa se identifica como tal.

## Reproducibilidad

Las aplicaciones funcionan sin red con datos y referencias incluidos en el
paquete. Exportan definiciones, muestra, unidades, estimaciones, diagnósticos
y figuras. La conversión de fuentes y regeneración de referencias son tareas
de desarrollo separadas de la ejecución normal. El software de referencia
no es una dependencia de ejecución.

La [guía del puente estructural](structural_bridge.md) explica las convenciones
de covarianza y la [guía de benchmarks](research_benchmarks.md), las fuentes,
comparaciones estrictas y controles negativos.
