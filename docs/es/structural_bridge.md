> 🇬🇧 [English](../structural_bridge.md) · 🇪🇸 Español

# De momentos empíricos a parámetros estructurales

`puremacro.structural` ajusta un modelo estructural, expresado como una función
de Python, a estimaciones de momentos empíricos. Conserva su covarianza
completa, etiquetas, unidades y metadatos de investigación. Conecta las
estimaciones con el modelo; la optimización por sí sola no lo identifica.

La covarianza debe ser **Cov(estimadores de los momentos empíricos)**. Su
diagonal contiene, por ejemplo, errores estándar al cuadrado de respuestas
estimadas, y fuera de la diagonal aparecen sus covarianzas. No se debe pasar
la covarianza de las observaciones originales ni una matriz multiplicada por
el tamaño de muestra. `fit_structural` no introduce ese multiplicador.

## Ejemplo completo

```python
import numpy as np
from puremacro.structural import MomentTargets, fit_structural

h = np.arange(4)
empirical_irf = np.array([0.80, 0.61, 0.47, 0.34])
covariance = 0.01 * 0.5 ** np.abs(h[:, None] - h[None, :])
targets = MomentTargets.from_irf(
    empirical_irf,
    covariance=covariance,
    response="output",
    shock="monetary",
    units="percent per 25bp tightening",
    metadata={"sample": "illustrative", "shock_size": 25, "shock_unit": "bp"},
)

def moments_at(theta):
    impact, persistence = theta
    return impact * persistence ** h

result = fit_structural(
    moments_at, targets, [0.7, 0.7],
    parameter_names=("impact", "persistence"),
    bounds=[(0, None), (0, 0.99)],
    moment_labels=targets.labels,
)
print(result.summary())
print(result.moment_fit())
print(result.to_latex())
```

Los datos del ejemplo son sintéticos. Las respuestas del modelo deben usar
las mismas variables, frecuencia de los horizontes, convención de acumulación,
transformaciones y normalización del choque que los momentos empíricos. El
adaptador registra estas decisiones, pero no las deduce ni las reconcilia.

## Construcción y combinación de momentos

| Interfaz | Contrato |
|---|---|
| `MomentTargets(values, covariance, labels, units=..., metadata=...)` | Vector de momentos, covarianza cuadrada completa y etiquetas únicas. |
| `MomentTargets.from_frame(frame, covariance=...)` | Columnas `label`, `estimate`, `unit`; permite cambiar sus nombres. Sin columna de etiquetas usa el índice; sin unidades registra `unspecified`. |
| `MomentTargets.from_irf(irf, covariance=..., response=..., shock=..., units=...)` | Vector de una respuesta a un choque, o tabla LP con `h` y `beta`. Se debe seleccionar explícitamente la respuesta de un tensor VAR. |
| `MomentTargets.from_draws(draws, values=..., labels=...)` | Filas de réplicas conjuntas de los estimadores; covarianza muestral con `ddof=1`, sin dividir entre el número de réplicas. |
| `targets.select(labels)` | Selecciona/reordena el vector y los dos ejes de la covarianza. |
| `targets.rescale(factors, units=...)` | Transforma conjuntamente `D m` y `D S D'`, registrando las unidades anteriores. |
| `targets.stack(other, cross_covariance=...)` | Requiere covarianza cruzada explícita o `assume_independent=True`. Conserva los metadatos de ambos componentes. |

Las bandas y errores estándar por horizonte no determinan una covarianza
conjunta. Cada réplica bootstrap debe volver a estimar **todos** los momentos
con un procedimiento válido para su dependencia temporal. Las muestras
posteriores bayesianas o los horizontes remuestreados por separado no son,
en general, entradas válidas para `from_draws`. `values` conserva las
estimaciones originales. No se descartan réplicas fallidas automáticamente.

Las entradas pandas etiquetadas deben coincidir exactamente con el orden
declarado. Los arrays son posicionales. Si el modelo devuelve una Series, se
comprueba su índice en cada evaluación; `moment_labels=` permite declarar
explícitamente el orden del modelo.

Los arrays se copian y quedan en modo de solo lectura. Los metadatos
estructurados como diccionarios, listas y arrays se copian y congelan.
Los adaptadores conservan los atributos de las tablas; combinar momentos
conserva ambas muestras y normalizaciones, sin convertir unidades implícitamente.

## Covarianza conjunta de proyecciones locales

`lp_moment_targets` estima las respuestas y su covarianza conjunta desde una
tabla temporal con observaciones equiespaciadas:

```python
from puremacro.structural import lp_moment_targets

import pandas as pd
# Datos sintéticos transformados para un ejemplo ejecutable de la API.
rng = np.random.default_rng(19)
data = pd.DataFrame(rng.normal(size=(160, 3)),
                    columns=["output", "inflation", "monetary_innovation"],
                    index=pd.period_range("1980Q1", periods=160, freq="Q"))
empirical = lp_moment_targets(
    data,
    responses=("output", "inflation"),
    shock="monetary_innovation",
    horizons=(0, 1, 2, 3),
    response_units={"output": "log points", "inflation": "percentage points"},
    shock_unit="percentage points",
    frequency="Q",
    lags=2,
    bandwidth=4,
    shock_size=0.25,
)
```

Como en `lp_hac`, la variable dependiente es **`y[t+h]-y[t-1]`**. Se incluyen
el choque contemporáneo, rezagos de respuesta y choque, y los controles
declarados tanto contemporáneos como rezagados. La respuesta estructural debe
coincidir con esa convención. El orden es primero respuesta, luego horizonte;
las etiquetas tienen formato `response:shock:h=H`.

Las contribuciones de influencia de los coeficientes se alinean en el índice
temporal original, con ceros fuera de cada ventana de estimación. Un único
núcleo Bartlett HAC conserva la covarianza entre respuestas y horizontes.
El ancho predeterminado es `max(horizons)+1`, una regla heurística explícita;
no se aplica corrección de muestra pequeña. Se requiere exogeneidad del
choque condicional en los regresores y condiciones temporales regulares para
HAC. No es un estimador LP-IV ni inferencia robusta a identificación débil.
Si demasiados momentos producen una covarianza singular, el ajuste la rechaza.

Los datos deben ser finitos y el índice único y creciente. Los índices de
fechas/períodos se verifican contra `frequency`; otros índices dependen de la
frecuencia declarada. No se eliminan períodos faltantes ni se convierten
porcentajes según nombres. Se conservan muestra, frecuencia, observaciones
por momento, transformación, ancho de banda, unidades y tamaño del choque.

## Criterio y covarianza local

Sean `S` la covarianza suministrada, `r(theta)=m(theta)-m_hat` y
`G=dm/dtheta`. El criterio es

\[
Q(\theta)=r(\theta)'Wr(\theta),\qquad W=S^{-1}\text{ por defecto}.
\]

Se factoriza `S=D L L' D`, donde `D` contiene los errores estándar y `L` es
el factor de Cholesky de la matriz de correlaciones. Los residuos ponderados
son `L^{-1}D^{-1}r`. Esta estandarización evita que las comprobaciones de
covarianza dependan de las unidades de los momentos. El objetivo reportado
es el criterio cuadrático completo con los pesos originales.

El optimizador usa coordenadas centradas en `theta0`, escaladas por el
jacobiano ponderado inicial, y escalamiento adaptativo de la región de
confianza. También divide todos los residuos por una escala común de
incertidumbre muestral. Se conservan los pesos relativos y el minimizador,
evitando convergencia prematura por unidades de parámetros, desplazamientos
grandes o múltiplos escalares de pesos personalizados. Estas normalizaciones
internas no modifican el objetivo reportado ni la covarianza paramétrica.

Para una matriz fija `weight=W`, positiva definida, se calcula

\[
A=(G'WG)^{-1}G'W,\qquad
\widehat{\operatorname{Cov}}(\hat\theta)=ASA'.
\]

Con ponderación óptima se reduce a `(G'S^{-1}G)^{-1}`. La implementación usa
una SVD del jacobiano ponderado y factores de covarianza, sin invertir las
ecuaciones normales. No estima una varianza residual ni modifica `S` según
la calidad del ajuste.

Las fórmulas requieren especificación correcta, suavidad, covarianza
consistente e identificación interior regular. El marco de mínima distancia
y el resultado de sobreidentificación se presentan en Newey y McFadden
(1994), secciones 3.1, 4.1, 5.2 y 9.5.
[Capítulo original](https://users.ssc.wisc.edu/~xshi/Newey_Mcfadden_handbook.pdf).

`jac(theta)` suministra el jacobiano **sin ponderar**, de momentos por
parámetros. En su ausencia se usan diferencias finitas de segundo orden,
unilaterales cerca de los límites. Esto no implementa derivadas faltantes de
distribuciones de agentes heterogéneos o del equilibrio general. Los momentos
estocásticos, no suaves o discontinuos por discretización requieren otro
diseño inferencial; fijar una semilla no elimina el error de simulación.
El método de región de confianza es local y no prueba optimalidad global.
[Referencia de least_squares de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## Diagnósticos y aceptación

`success` describe la optimización numérica. `inference_valid` indica que las
comprobaciones numéricas locales permiten usar las fórmulas condicionales
descritas. Ninguno certifica que el modelo económico sea correcto.

Los errores estándar e intervalos se devuelven como NaN, con advertencia y
razones identificadas, si falla la optimización, falta rango de columnas,
el condicionamiento supera `condition_limit`, algún parámetro se encuentra
en/cerca de un límite, o una solución interior incumple la condición numérica
de primer orden (`nonstationary_solution`). Esta última comprueba la proyección
del residuo sobre el espacio de columnas del jacobiano ponderado local;
un modelo sobreidentificado puede conservar residuos ortogonales a ese espacio.
Se conservan las estimaciones y los residuos.
El rango usa `s_i > rank_rtol*s_max` (`1e-10` por defecto); el límite de
condicionamiento es `1e8`. Este depende de las unidades de los parámetros.
Son diagnósticos numéricos, **no pruebas estadísticas de identificación débil**,
y el rango local no establece identificación global.

Las matrices deben ser finitas, simétricas y semidefinidas positivas.
El ajuste requiere además una factorización numéricamente positiva definida.
Se rechazan covarianzas singulares sin regularización, recorte de autovalores,
pseudoinversa ni eliminación automática de momentos. El investigador puede
usar `select` para retirar redundancias de forma explícita y documentada.

Los resultados del modelo no finitos, dimensiones incorrectas y derivadas
inválidas producen errores. Una solución económica fallida debe producir un
error dentro de `moments_at`; no se sustituye por un vector artificial o una
penalización automática.

## Sobreidentificación y momentos reservados

Con `q>p`, la referencia chi-cuadrado con `q-p` grados de libertad requiere
especificación correcta, identificación regular y ponderación óptima.
`assume_correct_specification=True` solicita la prueba y registra el supuesto;
no comprueba que este sea verdadero. `j_statistic` y `j_pvalue` son NaN con
ponderación personalizada, identificación exacta/insuficiente o diagnósticos
numéricos fallidos. `objective` siempre reporta el criterio; `j_df=max(q-p,0)`.

```python
unused = MomentTargets([0.2], [[0.02]], ("output_later",), targets.units[0])
checked = fit_structural(
    moments_at, targets, [0.7, 0.7],
    bounds=[(0, None), (0, 0.99)],
    held_out=unused,
    held_out_moments_at=lambda theta: [theta[0] * theta[1] ** 6],
)
print(checked.moment_fit(held_out=True))
```

Las etiquetas reservadas deben diferir de las usadas en el ajuste. No afectan
el objetivo. La tabla muestra momentos, errores estándar de los objetivos,
predicciones y residuos, sin valor p ni intervalo predictivo: falta especificar
la incertidumbre paramétrica conjunta y su covarianza con esos estimadores.
Reservar momentos de la misma muestra no implica independencia.

## Evidencia y alcance

`tests/test_structural_bridge.py` comprueba una solución GLS cerrada con
covarianza correlacionada, covarianza sándwich con ponderación arbitraria,
escala de J, recuperación no lineal analítica, invariancia de unidades,
adaptadores y contratos de fallo. Son verificaciones analíticas y de software,
no una replicación empírica ni evidencia de cobertura nominal para todo
estimador posible. Quedan fuera las derivadas de distribuciones HA,
inferencia robusta a identificación débil, correcciones del ruido de
simulación, búsqueda global e inferencia bajo especificación incorrecta.
