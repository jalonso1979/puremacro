> 🇬🇧 [English](../ADVISORY.md) · 🇪🇸 Español

# Avisos de corrección

Se emite un aviso de corrección cuando una versión publicada de
puremacro devolvió un **número equivocado** — no un fallo, no una función
ausente, sino una respuesta bien formada y falsa. La distinción importa
porque un fallo se denuncia solo y un número equivocado no: acaba en un
cuadro, en una figura, en un dictamen.

Cada aviso nombra las versiones afectadas, la condición exacta bajo la
cual el error **se anula** (para que pueda descartar su propia estimación
sin volver a correrla) y qué hacer.

---

## 2026-10-02 — `solve_trade_equilibrium(method="quasi_condensed")` informaba `converged=True` con un residuo grande, versiones 4.2.0 a 4.4.0

**Corregido después de 4.4.0** (véase la sección Unreleased de `CHANGELOG.md`).
La rama `quasi_condensed` de `solve_trade_equilibrium` tomaba la bandera de
convergencia del solver cuasi-condensado (flexible), pero el residuo, los
flujos y los precios del posprocesamiento heredado, evaluado con la convención
de precios de MATLAB del solver (`replicate_matlab_precedence=True`), que la
solución cuasi-condensada no usa. La bandera y el residuo informado no
concordaban. En la calibración 2x2 del cuaderno 62 con un arancel del 25% y
`tol=1e-10`, 4.4.0 devolvía `converged=True` con `max_residual` 0.0487
(`residual_norm` 0.155); con `sigma_y=0.5`, 0.498. Ahora el residuo es el que
juzgó la bandera, los flujos se posprocesan con la convención cuasi-condensada
y coinciden hasta 1e-8 con una solución Newton heredada del mismo modelo
(`replicate_matlab_precedence=False`), y las opciones flexibles activas
lanzan `ValueError`, porque los campos de flujos heredados de
`TradeEquilibriumResult` no pueden describirlas.

| Superficie | Condición afectada | No afectada cuando | Recomendación |
|---|---|---|---|
| `solve_trade_equilibrium(..., method="quasi_condensed")` | cualquier llamada | — | `x_sol` era la solución cuasi-condensada y la bandera era correcta para ese modelo; los campos de residuo y los flujos no. Vuelva a correr con la versión corregida, o use `solve_flexible_trade_equilibrium(..., method="quasi_condensed")`. |
| ídem, con `sigma_y > 0` o un `config` activo | flujos, precios, IPC, términos de intercambio | — | Describen el modelo heredado, no el resuelto; tome los flujos de `metadata` de `solve_flexible_trade_equilibrium`. |
| ídem, con `fiscal_closure`/`recycling_params`/`sigma`/`capacity_margins` | la solución ignoraba la opción | la opción estaba en su valor por defecto | El equilibrio devuelto no resuelve el modelo pedido. |

`solve_flexible_trade_equilibrium` nunca usó esta rama; sus `converged` y
`residual_norm` eran coherentes.

### ¿Le afecta?

```python
res = solve_trade_equilibrium(calib, ..., method="quasi_condensed")
res.converged and res.max_residual > res.metadata["tol"]   # True → bandera y residuo discrepaban
```

### Qué hay que volver a correr

- Cualquier cuadro construido con los flujos, precios, IPC o términos de
  intercambio de un resultado de
  `solve_trade_equilibrium(..., method="quasi_condensed")`.

---

## 2026-10-02 — Las propiedades de flujos de `FlexibleTradeEquilibriumResult` ignoraban el modelo resuelto, versiones 4.2.0 a 4.4.0

**Corregido después de 4.4.0** (véase la sección Unreleased de `CHANGELOG.md`).
Las propiedades `exports`, `imports`, `cpi`, `terms_of_trade`, `gdp` y
`gdp_fc` de un resultado de `solve_flexible_trade_equilibrium` evaluaban las
ecuaciones de flujos heredadas Cobb-Douglas/Leontief en `x_sol` sin aranceles
y con la convención de precios de MATLAB, fuera cual fuera la llamada. Medido
en 4.4.0, calibración 2x2 del cuaderno 62, arancel del 25% a los insumos
intermedios del país B:

- ruta heredada (configuración por defecto): exportaciones `[60.73, 68.94]`
  frente a `[60.73, 69.19]` de `solve_trade_equilibrium` con los mismos
  argumentos;
- ruta cuasi-condensada, configuración por defecto: exportaciones
  `[55.87, 63.12]` frente a `[56.73, 64.26]` de una solución Newton heredada
  del mismo modelo;
- ruta cuasi-condensada, `sigma_trade=5`: exportaciones `[58.20, 59.86]`
  frente a los flujos bilaterales resueltos `[62.15, 58.26]`.

`cpi`, `terms_of_trade` y `gdp`/`gdp_fc` coincidían en los dos casos con
configuración por defecto (dependen del estado, no de las cantidades que
dependen del arancel); con una configuración activa el IPC y los términos de
intercambio heredados describen otro modelo, y con `variable_markups=True`
`gdp` omitía los beneficios de margen. Las propiedades describen ahora el
modelo resuelto; `cpi`/`terms_of_trade` lanzan `NotImplementedError` con una
configuración activa. 4.4.0 documentaba la mitad cuasi-condensada como
problema conocido; la mitad de la ruta heredada no se conocía.

| Superficie | Condición afectada | No afectada cuando | Recomendación |
|---|---|---|---|
| `exports`, `imports` | cualquier `tau`, `tau_fd`, `tauf`, `tauf_fd`; cualquier configuración activa en la ruta cuasi-condensada | sin aranceles y con la configuración por defecto en la ruta heredada | Vuelva a correr, o sume filas/columnas de `metadata["bilateral_trade"]` (ruta cuasi-condensada). |
| `cpi`, `terms_of_trade` | configuración flexible activa | configuración por defecto | No hay índice sustituto para el modelo flexible; use los flujos de `metadata`. |
| `gdp` | `variable_markups=True` | sin márgenes | Sume `metadata["markup_profits"]`, o use `metadata["household_income"]`. |
| `welfare_decomposition` | — | siempre: conserva su aproximación histórica y sus valores | Sin cambios. |

### ¿Le afecta?

```python
res = solve_flexible_trade_equilibrium(calib, ...)
tariffs = any(v is not None for v in res.metadata["tariff_inputs"].values())
tariffs or bool(res.metadata["active_flexible_settings"])   # True → vuelva a correr
```

### Qué hay que volver a correr

- Volúmenes de comercio, balanzas comerciales y efectos de términos de
  intercambio leídos de estas propiedades en escenarios arancelarios o con
  opciones flexibles activas.

---

## 2026-10-02 — Grados de libertad de la posterior de Σ en `minnesota_gibbs`, versiones 0.92.0 a 4.4.0

**Corregido después de 4.4.0** (véase la sección Unreleased de `CHANGELOG.md`).
Bańbura, Giannone y Reichlin (2010; ECB WP 966, p. 12, ec. 7) añaden la
previa impropia `|Ψ|^-(n+3)/2` a las observaciones ficticias, lo que da
`Ψ | Y ~ iW(Σ̃, T_d + 2 + T - k)`. `minnesota_gibbs` usaba `T_d + T - k`. La
media posterior de Σ, `Σ̃ / (ν - n - 1)`, era por tanto demasiado grande en el
factor `(ν - n + 1)/(ν - n - 1)`: 5.3% en un VAR(1) de 3 variables con 39
observaciones, 1.0% con 3 variables, 4 rezagos y 200 trimestres. Los
coeficientes simulados heredan la escala a través de `Σ ⊗ (X*'X*)⁻¹`, así que
su dispersión posterior era demasiado amplia en cerca de la mitad de ese
porcentaje. La media posterior de los coeficientes (`A_mean`,
`intercept_mean`), `minnesota_posterior` y la verosimilitud marginal de
`minnesota_optimal_lambda` no se ven afectadas.

| Superficie | Condición afectada | No afectada cuando | Recomendación |
|---|---|---|---|
| `Sigma_draws`, `nu_post` y bandas posteriores de `A_draws`/`intercept_draws` de `minnesota_gibbs` | siempre | estimaciones puntuales de `A_mean`/`intercept_mean` | Vuelva a correr; el efecto decrece como `2/(T + T_d - k - n)`. |

### ¿Le afecta?

```python
T, n = Y.shape                              # sus datos y el orden de rezagos p
nu_old = (T - p) + (n * p + n + 1) - (n * p + 1)   # T + T_d - k que usaba 4.4.0
(nu_old - n + 1) / (nu_old - n - 1) - 1     # sobreestimación relativa de E[Σ | Y]
```

### Qué hay que volver a correr

- Bandas de credibilidad de respuestas al impulso, pronósticos y
  descomposiciones de varianza calculadas con simulaciones de
  `minnesota_gibbs`, sobre todo con muestras cortas o sistemas grandes.

---

## 2026-10-02 — Los solvers de splines y Smolyak ignoraban `gamma` y resolvían utilidad logarítmica en sus métodos de Bellman, versiones 3.3.0 a 4.4.0

**Corregido después de 4.4.0** (véase la sección Unreleased de `CHANGELOG.md`).
Tres errores relacionados en el modelo de crecimiento incorporado de
`puremacro.vfi.splines` y `puremacro.vfi.smolyak`, verificados con la ecuación
de Euler del modelo CRRA escrita a mano:

1. Ambos leían solo `params["sigma"]`. `params={"gamma": 2}` (la grafía de
   `CollocationProblem`, `FEMProblem` y los motores discretos) resolvía en
   silencio utilidad logarítmica: residuo de Euler 0.151 (splines) y 0.0345
   (Smolyak, 2 capitales) en la política devuelta, que informaba
   `converged=True`.
2. El solver de Bellman con splines (`method="bellman"`) usaba utilidad
   logarítmica con cualquier curvatura: residuo 0.151 con `sigma=2`.
3. `solve_smolyak(method="bellman")` iteraba sobre la política cerrada
   `k'_m = alpha_m beta Y` de utilidad logarítmica con depreciación total y la
   devolvía con `converged=True` para cualquier `sigma`, `gamma`, `delta` o
   `return_fn` (residuo 0.0345 con `sigma=2`). Ahora lanza
   `NotImplementedError` fuera de ese caso.

| Superficie | Condición afectada | No afectada cuando | Recomendación |
|---|---|---|---|
| splines y Smolyak con `method="euler"` | `params` tiene `gamma` y no `sigma` | se da `sigma`, o curvatura 1 | Vuelva a correr con la versión corregida, o pase `sigma`. |
| splines con `method="bellman"` | `sigma` (o `gamma`) ≠ 1 | utilidad logarítmica | Vuelva a correr con la versión corregida. |
| `solve_smolyak(method="bellman")` | `sigma`/`gamma` ≠ 1, `delta` ≠ 1 o un `return_fn` | utilidad logarítmica con depreciación total | Use `method="euler"`. |

### ¿Le afecta?

```python
p = problem.params
curv = p.get("sigma", p.get("gamma", 1.0))
("gamma" in p and "sigma" not in p and curv != 1.0) \
    or (problem.method == "bellman" and (curv != 1.0 or p.get("delta", 1.0) != 1.0))   # True → vuelva a correr
```

### Qué hay que volver a correr

- Políticas, funciones de valor, valores marginales y gradientes IFT de estos
  solvers con utilidad no logarítmica (y, para Bellman con Smolyak,
  depreciación parcial).

---

## 2026-10-02 — `vif` con datos en niveles y constante, versiones 2.6.0 a 4.3.0

**Corregido en 4.4.0.** `inference.collinearity.vif` reproducía
`variance_inflation_factor` de statsmodels operación por operación, incluidas
sus regresiones auxiliares sobre los niveles sin centrar. Cuando la media de
los regresores es grande respecto a su dispersión, esas regresiones sufren una
cancelación catastrófica y el VIF pierde dígitos, o todos. Con una columna
constante explícita el VIF es exactamente invariante a desplazar cualquier
columna, así que las columnas no constantes se calculan ahora con datos
centrados; el resultado coincide con la aritmética racional exacta hasta
`0.33 * eps * max VIF` en diseños de prueba con medias de hasta `1e9`.

Medido en 4.3.0 (statsmodels devuelve los mismos números): tres regresores
normales estándar desplazados a media `1e7` daban 1.007122 frente a un exacto
de 1.007165 (`4e-5` relativo); una tasa de interés junto a un PIB en yenes
(escala `2e15`) daba 0.053, imposible porque todo VIF es al menos 1 (exacto:
1.005); y diseños aleatorios desplazados erraban hasta en 99%.

| Superficie | Condición afectada | No afectada cuando | Recomendación |
|---|---|---|---|
| `vif` | `exog` tiene una columna constante explícita y regresores cuya media es grande respecto a su desviación estándar (datos en niveles) | los regresores tienen media cercana a cero o se centran antes de la llamada; el VIF de la propia columna constante; diseños sin columna constante | Vuelva a correr con 4.4.0. En los diseños afectados los valores ahora difieren de statsmodels, que conserva el error. |

### ¿Le afecta?

```python
X  # su diseño, con una columna constante
abs(X[:, 1:].mean(axis=0) / X[:, 1:].std(axis=0)).max()   # por encima de ~1e5 → vuelva a correr
```

El error de 4.3.0 crece con esa razón: por debajo de `4e-12` relativo hasta
`1e5`, `3e-8` en `1e6`, `1e-4` en `1e7`, y más cuando los regresores además
son casi colineales o tienen tendencia (el ejemplo del PIB arriba).

### Qué hay que volver a correr

- **Cualquier diagnóstico de colinealidad con datos en niveles** (PIB, índices
  de precios, población) calculado con `vif` y un intercepto. Los datos
  centrados o en tasas de crecimiento no se vieron afectados.

---

## 2026-09-30 — choques de margen de Smets-Wouters (2007), datos incluidos y objetivos de replicación, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0** (véase la sección Unreleased de `CHANGELOG.md`).
Lo encontró la revisión de cuadernos del 30 de septiembre, como todas las
entradas fechadas el 2026-09-30. En la evidencia de Smets-Wouters (2007)
coincidían tres errores distintos.

1. En el modelo programado a mano (`dsge.smets_wouters`), los procesos de
   margen de precios y salarios eran `spinf_t = crhopinf*spinf_{t-1} + epinf_{t-1}`
   y lo mismo para `sw`/`ew`: sin término MA y con un trimestre de retraso.
   SW07 (ECB WP 722, pp. 14-15) y el `.mod` de Pfeifer incluido tienen
   `+ epinf_t - cmap*epinf_{t-1}`. Por eso `cmap` y `cmaw` nunca entraban en el
   modelo, `estimate_sw07` devolvía su distribución a priori como posterior, y
   las respuestas a `epinf`/`ew` eran erróneas incluso con `cmap = cmaw = 0`.
   Con la calibración del `.mod`, la respuesta de la inflación a un choque
   unitario `epinf` en los trimestres 0-2 era 0.932, 1.416, 0.255; es 1.177,
   0.307, -0.060. Los otros cinco choques eran correctos.
2. El `_sw07_data.csv` incluido medía las horas como `log(HOANBS/pop)`, 1/100
   del `100*log(horas medias NFB x empleo / pob)` de SW07, y el consumo y la
   inversión como PCE encadenado e inversión con existencias, en lugar de PCE
   nominal e inversión fija deflactados por el deflactor del PIB (correlación
   con las series del propio SW: horas 0.950, crecimiento de la inversión
   0.624). Toda verosimilitud sobre los datos incluidos está afectada, tanto con
   el modelo nativo como con `sw07_pfeifer.mod`.
3. Los objetivos de replicación -1673.72, -1686.09 y -2524.36 (log posterior y
   verosimilitudes marginales de Laplace y de media armónica) eran salidas de
   puremacro citadas como cuadros 1 y 2 de SW07, y el caso de la "moda"
   comparaba la mejor de 200 extracciones MCMC con las **medias** a posteriori
   del cuadro 1a. Además, los casos leían su fixture de `tests/fixtures`, así
   que fallaban en una rueda instalada (desde 2.6.0).

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `solve_sw07`, `smets_wouters._shock_irf` y todo lo construido sobre ellos | Respuestas, FEVD o descomposiciones con `epinf` o `ew` | Los otros cinco choques (`ea`, `eb`, `eg`, `eqs`, `em`) | Volver a correr. Las respuestas de los siete choques coinciden ahora con la solución del `.mod` hasta 1e-12. |
| `estimate_sw07`, `sw07_observation.make_state_space` | Siempre: cambiaron el modelo y los datos | Nunca | Volver a estimar. No compare log posteriores ni verosimilitudes marginales antiguas con las nuevas. |
| `_sw07_data.csv` con `sw07_pfeifer.mod` (cuadernos 41 y 42) | Cualquier estimación, suavizado, descomposición de choques o verosimilitud marginal sobre los datos incluidos | Sus propios datos | Volver a correr con el archivo reconstruido. |
| Casos de replicación `dsge_estimation` (2.6.0 a 4.3.0) | Siempre | Nunca | Los objetivos se etiquetan ahora como valores de regresión de puremacro (-822.04, -902.83, -908.03); el caso de la moda compara una moda optimizada con la columna Mode del cuadro 1 (13 parámetros, todos a menos de 6.2%). |

### ¿Le afecta?

```python
import pandas as pd
from importlib.resources import files
d = pd.read_csv(files("puremacro.dsge") / "_sw07_data.csv", comment="#")
d["log_hours"].std()   # cerca de 0.046: los datos de 4.3.0; cerca de 2.9: los reconstruidos
```

Cualquier número de SW07 calculado con los datos de 4.3.0, y cualquier
respuesta de SW07 a un choque de margen del modelo nativo, está afectado.

### Qué hay que volver a correr

- **Cualquier estimación, verosimilitud marginal, suavizado o descomposición de
  choques de SW07** sobre los datos incluidos, nativa o con el `.mod`.
- **Cualquier respuesta, FEVD o descomposición de `solve_sw07`** con `epinf` o
  `ew`.
- **Cualquier cita de -1673.72, -1686.09 o -2524.36 como resultados de SW07.**
  Con las correcciones, la moda a posteriori optimizada sobre los datos
  incluidos tiene log posterior -822.04 y log densidad marginal de Laplace
  -902.83. El -905.8 del propio SW07 (cuadro 2) usa una a priori con muestra de
  entrenamiento 1956-65 y no es un objetivo comparable.
- **En este repositorio:** los cuadernos 41 y 42 (inglés y español),
  `docs/replication.md` y el borrador para JOSS citan los números afectados.

---

## 2026-09-30 — variables locales de modelo de `.mod` congeladas en la calibración, versiones 2.0.0 a 4.3.0

**Corregido después de 4.3.0.** Una variable local de modelo de un `.mod`
definida sólo con parámetros (`#cbeta = 1/(1+constebeta/100);`) se evaluaba
una vez, con la calibración del archivo, y el número se insertaba en las
ecuaciones (desde 2.7.0; el lector por expresiones regulares, el único de
2.0.0 a 2.6.0 y una alternativa de reserva desde entonces, la guardaba como un
parámetro ficticio). Dynare sustituye la expresión. Todo
cálculo que vuelve a resolver el modelo con otros valores de los parámetros
usaba por tanto un modelo híbrido: se movían los parámetros que aparecen
directamente en las ecuaciones, pero no las variables locales construidas con
ellos.

En `sw07_pfeifer.mod`, `constebeta` entra en el modelo sólo a través de
variables locales, así que su verosimilitud era exactamente plana y su
posterior igual a su a priori; `csigma`, `ctrend`, `constepinf`, `calfa`, `cg`,
`ctou` y `clandaw` entraban sólo en parte. Volver a resolver con
`csigma = 2.5` daba sd(y) = 30.1307 en lugar de 29.9343 (una carga nueva). Con
los datos incluidos actuales, la log-verosimilitud en los valores iniciales de
`estimated_params` es -1766.35, no -1776.93, y `constebeta` en 0.5 / 1.0 / 1.5
da -1749.77 / -1794.92 / -1881.05, donde los tres daban -1776.93.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `load_mod(..., params=)`, `osr` y búsquedas de política óptima, widgets de parámetros, SMC, gradientes analíticos y NUTS, `identification()`, `LinearModel.estimate()` / `estimate_dsge` sobre un modelo `.mod` | El modelo tiene una variable local `#` cuyo lado derecho sólo contiene parámetros y números, y el cálculo movió uno de esos parámetros | Resultados con la propia calibración del archivo; variables locales con variables endógenas; modelos escritos como funciones de residuos en Python; `estimate_sw07` programado a mano | Volver a correr con la versión corregida. |
| `load_mod(..., params=)` con un bloque `steady_state_model` o `initval` | El bloque depende de un parámetro modificado | Sin esa dependencia | Lanzaba `SteadyStateError`; ahora reevalúa el bloque con los nuevos valores. |
| Nombres de parámetros o variables que son palabras clave de Python (`lambda`, `yield`) | Siempre | Otros nombres | Lanzaban `SyntaxError` o `ValueError`; ahora se analizan. |

### ¿Le afecta?

Busque en el bloque `model;` una línea que empiece con `#` y cuyo lado derecho
sólo contenga parámetros y números. Si la hay, y corrió algún cálculo de la
primera fila con valores de los parámetros distintos de la calibración del
archivo, le afecta.

### Qué hay que volver a correr

- **Cualquier estimación, análisis de identificación, búsqueda `osr`,
  gradiente o resultado de estática comparativa** obtenido de un `.mod` así.
- **En este repositorio:** los cuadernos 41 y 42 y la lección T02_E del curso
  (`sw07_pfeifer.mod`).

---

## 2026-09-30 — `identification()` evaluada en una mezcla de la calibración y el punto pedido, versiones 2.6.0 a 4.3.0

**Corregido después de 4.3.0.** `dsge.identification` y
`LinearModel.identification` calculaban todas las cantidades reportadas
(rangos, `is_identified`, valores singulares, números de condición,
combinaciones del espacio nulo, colinealidad, fuerza de Ratto) a partir de una
mezcla de la calibración y el punto que pedía el usuario. Cada columna
estructural movía sólo su propio parámetro fuera de la calibración; las
columnas de choques y de errores de medida usaban la solución calibrada, con
`SE_`/`CORR_` dados sólo por nombre tomados en 1 y 0 en lugar de los valores
declarados; y la diferencia unilateral de reserva restaba la solución
calibrada (en un AR(1) devolvía -78,594.7 donde el valor verdadero es 1.78).
`prior_mc` perturbaba los valores base en un 5% en lugar de extraer de las
distribuciones a priori. Además, `build_dynare` detecta numéricamente las
variables predeterminadas en la calibración, así que un rezago calibrado en 0
desaparecía de toda nueva resolución y toda derivada respecto de su
coeficiente valía exactamente cero.

Parámetros realmente no identificados podían reportarse como identificados (un
par `k1*k2` en un `.mod` cuyas medias a priori difieren de la calibración
mostraba rangos 2/2/2/2 en lugar de 1/1/1/1), y parámetros identificados como
no identificados: `crhoms`, `crhopinf`, `crhow`, `cmap` y `cmaw` de SW07,
calibrados en 0, tenían sensibilidad exactamente 0, con rangos 29/31/29/29 de
36 en la llamada por defecto en lugar de 34/36/34/34.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `identification()` con `params={...}`, `p_dict`, `EstimatedParams` o una lista de nombres sobre un `.mod` con `estimated_params` | El punto analizado (valores, INITVAL o medias a priori) difiere de la calibración | El punto es la calibración y no se aplica ninguna de las filas siguientes | Volver a correr. |
| La llamada por defecto sobre un `.mod` con `estimated_params` | Los INITVAL o las medias a priori difieren de la calibración | Coinciden con ella | Volver a correr. |
| `prior_mc > 0` | Siempre | Nunca | Volver a correr; las tasas de identificación no se basaban en las a priori. |
| Un parámetro en una cota declarada, o a menos de un paso de diferencias finitas de ella | Siempre | Lejos de las cotas | Volver a correr. |
| Modelos `build(..., linearize="level")` | Un estado estacionario positivo, incluso en la calibración | Modelos log-linealizados o con estado estacionario cero | Volver a correr. |
| `SE_<shock>`, `CORR_<s1>_<s2>` o `ME_<obs>` dados por nombre | La covarianza declarada de los choques no es la identidad, o hay error de medida declarado, incluso en la calibración | Covarianza identidad, sin error de medida | Volver a correr; los `SE_`/`ME_` por nombre usan ahora los valores declarados. |
| Modelos `build_dynare` / `load_mod` | Un coeficiente de rezago vale 0 en la calibración y un parámetro analizado, o el punto pedido, lo vuelve distinto de cero | Ningún rezago calibrado en 0 | Volver a correr. |
| `LinearModel.estimate(check_identification=True/"raise")` | Cualquiera de las condiciones anteriores en el punto de partida de la estimación | Como arriba | La comprobación previa podía dejar pasar un modelo no identificado o fallar sin motivo. |

### ¿Le afecta?

Sólo eran correctos los resultados calculados en la calibración, sin
`prior_mc`, lejos de las cotas y sobre un modelo no cubierto por las cuatro
últimas filas del cuadro. En otro caso, vuelva a correrlos con la versión
corregida. Tenga en cuenta que un punto pedido sin solución estable única
lanza ahora `ValueError`.

### Qué hay que volver a correr

- **Cualquier análisis de identificación en los casos anteriores**, incluidos
  los rangos y las ordenaciones por fuerza reportados para Smets-Wouters (2007).
- **En este repositorio:** el cuaderno 45 llama a `identification()` en la
  calibración, donde el defecto no actúa; sus números no cambian.

---

## 2026-09-30 — HANK de dos activos en el espacio de secuencias, versiones 3.1.0 a 4.3.0

**Corregido después de 4.3.0.** Los resultados del solver de dos activos
anteriores a esta corrección no son válidos. La iteración de los hogares nunca
convergía: se quedaba en un ciclo de periodo 2 (cambio en norma del supremo
0.39) mientras `converged` valía siempre True. Con la calibración por defecto,
β(1+r_a) = 1.0146 > 1, así que toda la riqueza ilíquida se acumulaba en el
techo de la malla a_max = 30. Los jacobianos Fake-News no tenían términos de
anticipación (su triángulo superior estricto era cero) ni el desplazamiento ex
ante de fecha para r_b, y se perdía riqueza donde se recortaba b' > b_max. La
respuesta de equilibrio general a un choque monetario dependía por tanto del
horizonte: tras un recorte de 25 pb, el consumo en el impacto era -0.0038 con
T = 16, +0.0014 con T = 20, -0.0045 con T = 30 y -0.0027 con T = 40, y las
respuestas no se amortiguaban.

Con la corrección, las mismas llamadas dan +0.0031 en todo horizonte de 16 a
300 y las respuestas se amortiguan; el estado estacionario tiene A = 4.59,
B = 0.61 y una proporción de hogares que viven al día de 0.39, sin masa en
a_max. Cambian los valores por defecto (β de 0.985 a 0.98, `r_b_ss` de 0.01 a
0.005, `r_a_ss` de 0.03 a 0.0125, `chi_0` de 0.25 a 1.0, `a_max` de 30 a 40,
`b_max` de 15 a 10, nuevo `a_bar = 0.25`), `steady_state["Y"]` es C + CHI, y
los jacobianos son los del sector hogares cerrado con la regla de ingreso
z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}, con la que la riqueza total A + B
es constante en equilibrio.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `solve_two_asset_hank_sequence_space` | Siempre | Nunca | Volver a correr con la versión corregida; su calibración es otra, así que compare formas, no niveles. |
| `solve_hank_bridge` sobre `hank_two_asset.mod`, `HANKModel` con `assets = 2` | Siempre | Nunca | Volver a correr. |
| Solver de un activo (`solve_hank_sequence_space`, `hank_ssj.mod`) | — | Siempre: sus resultados son idénticos bit a bit | Nada que hacer. |

### ¿Le afecta?

Cualquier número de las tres superficies de dos activos de arriba en
3.1.0-4.3.0. Un estado estacionario de riqueza ilíquida igual a `a_max`, o una
respuesta cuyo signo cambia con el horizonte `T`, es la señal.

### Qué hay que volver a correr

- **Cualquier respuesta, jacobiano o estado estacionario del HANK de dos
  activos.**
- **En este repositorio:** el cuaderno 46 (inglés y español) imprimía un
  consumo en el impacto de -0.0038 mientras su texto dice que sube; la llamada
  da ahora +0.0031. El solver corregido es un modelo docente estilizado,
  contrastado con jacobianos por fuerza bruta, identidades presupuestarias e
  invariancia al horizonte, no con resultados publicados de dos activos.

---

## 2026-09-30 — estimación de VAR con cambio de régimen de Markov, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** El paso M del EM de `var.regime.ms_var_fit`
actualizaba la matriz AR común por MCO sin ponderar, ignorando las covarianzas
de cada régimen. Cuando las varianzas de los regímenes difieren, ese paso no
maximiza el objetivo del EM: la log-verosimilitud podía bajar de una iteración
a otra y el ajuste se detenía por debajo del estimador de máxima verosimilitud
mientras reportaba `converged=True`. Una cresta absoluta de 1e-8 sobre cada
covarianza de régimen hacía además que las estimaciones dependieran de las
unidades de los datos. El docstring afirmaba que la función reproduce Hamilton
(1989); estima otro modelo, con intercepto y varianza cambiantes (MSIH) y una
matriz AR común.

Con los datos de crecimiento del PNB de Hamilton (K = 2, p = 4), la llamada por
defecto devolvía loglik -180.48, sin converger; su régimen de intercepto bajo
tenía un intercepto positivo (0.31), P(permanecer) = 0.87 y cubría 66 de 131
trimestres. El ajuste corregido da loglik -179.16, interceptos 1.21 y -0.06,
P(permanecer en el bajo) = 0.79 y 35 trimestres de media baja. Con una
variable en unidades 1e-4 veces menores, el ajuste antiguo perdía 173 puntos
de log-verosimilitud.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Estimaciones de `ms_var_fit`, probabilidades filtradas y suavizadas, y `girf` construido sobre ellas | Varianzas de régimen distintas, o una variable con varianza absoluta pequeña (por ejemplo tasas en decimales) | Varianzas de régimen casi iguales y variables de escala unitaria (el ejemplo de ciclo económico incluido cambia poco: acierto de 98.0% a 97.2%) | Volver a estimar; revise `loglik_path`, `converged` y `sigma_at_floor`. |
| Afirmaciones de que `ms_var_fit` reproduce Hamilton (1989) | Siempre | — | Para el cuadro I de Hamilton use `MarkovAutoregression(k_regimes=2, order=4, switching_ar=False)` de statsmodels. |

### ¿Le afecta?

Vuelva a ajustar con la versión corregida y compare `loglik`: un valor mayor
significa que la estimación antigua no era el máximo. En la versión antigua,
un resultado con `converged=False`, o con `n_iter` igual al tope (100 por
defecto), indica que se detuvo antes de tiempo.

### Qué hay que volver a correr

- **Cualquier datación de regímenes, matriz de transición o respuesta
  dependiente del régimen de `ms_var_fit`.**
- **En este repositorio:** el cuaderno 16 y el ejemplo `ms_var_business_cycle`.

---

## 2026-09-30 — errores estándar de Windmeijer en dos etapas para paneles dinámicos, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** `dynpanel.ab_gmm` y `dynpanel.bb_gmm` con los
valores por defecto `two_step=True, windmeijer=True` calculaban sus errores
estándar con la derivada de la matriz de pesos evaluada en los residuos de la
segunda etapa. Windmeijer (2005) y Stata usan los residuos de la primera etapa.
Los coeficientes no estaban afectados. Aparte, `collapse=False`, que es la
disposición por defecto de Stata, lanzaba `LinAlgError` ("step1 weight Z'HZ:
matrix is not positive definite") en todo modelo con dos o más rezagos de la
variable dependiente o con regresores exógenos rezagados, porque se conservaba
una columna de instrumentos toda en cero.

En el ejemplo abdata de Stata ([XT] xtabond, ejemplo 4), quitando a mano la
columna muerta, los errores estándar antiguos eran entre 1.4% y 68% demasiado
grandes (`w`: 0.2602 frente a 0.1546), así que ahí la inferencia era demasiado
conservadora. Con la disposición colapsada por defecto el error suele quedar
por debajo del 2% y va en cualquier dirección (antiguo frente a corregido:
0.554759 frente a 0.564362 para `ab_gmm` y 0.084758 frente a 0.082478 para
`bb_gmm` en paneles simulados; hasta cerca de 4% en abdata). El `ab_gmm`
corregido reproduce todos los coeficientes y errores estándar publicados de los
ejemplos 1, 2 y 4 de xtabond.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Errores estándar, estadísticos z, valores-p e intervalos de `ab_gmm`, `bb_gmm` | `two_step=True` y `windmeijer=True` (los valores por defecto) | `two_step=False`, o `windmeijer=False`; los coeficientes estimados en todos los casos | Volver a correr y sustituir los errores estándar reportados. |
| `diagnostics.windmeijer_correction` llamado directamente | Sin el nuevo `residuals_step1=` | Con él | Pase los residuos de la primera etapa; la llamada antigua advierte. |
| `collapse=False` | `lag_dep_var >= 2`, regresores exógenos rezagados o huecos | Otras disposiciones | Lanzaba un error; ahora poda las columnas muertas, como Stata. |

### ¿Le afecta?

Toda estimación en dos etapas con la corrección de Windmeijer, es decir, toda
llamada con los valores por defecto. Sólo cambian los errores estándar y lo que
se construye sobre ellos.

### Qué hay que volver a correr

- **Estimaciones de panel dinámico en dos etapas:** sustituya sus errores
  estándar. Para coincidir con `xtabond` de Stata, use `collapse=False` (y
  `two_step=False, robust=False` para su VCE homocedástica de una etapa).

---

## 2026-09-30 — a priori de Minnesota mal centrada con λ₂ ≠ 1, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** En `var.bvar._build_minnesota_dummies`, las
observaciones ficticias que usan `minnesota_gibbs`, su verosimilitud marginal
y `minnesota_optimal_lambda` escribían cada celda (rezago, variable) una vez
por ecuación, y ganaba la última ecuación. Con el valor por defecto λ₂ = 0.5,
el primer rezago propio de toda variable salvo la última de la lista se
centraba en 0.5 en lugar de 1 (el paseo aleatorio), con desviación típica a
priori λ₁λ₂ en lugar de λ₁; la última variable ignoraba λ₂ en las ecuaciones
cruzadas; y los resultados dependían del orden de las columnas. En el límite de
a priori muy ajustada (λ₁ = 1e-4), `minnesota_gibbs` devolvía
A1 = diag(0.5, 0.5, 1.0) en lugar de I. La a priori conjugada normal-Wishart
inversa no puede representar λ₂ ≠ 1: Bańbura, Giannone y Reichlin (2010) lo
imponen "bajo la condición θ = 1". El caso de validación estaba fijado en
λ₂ = 1, el único valor en que el error no puede verse.

El bloque es ahora la ec. (5) de BGR; las rutas NIW usan λ₂ = 1 por defecto y
cualquier otro valor advierte y usa 1. En un VAR(1) de 3 variables con T = 200
y los valores por defecto, la media posterior de Gibbs de A1[0,0] pasa de 0.501
a 0.545.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `minnesota_gibbs`, `minnesota_optimal_lambda` y su verosimilitud marginal | λ₂ ≠ 1, lo que incluye el antiguo valor por defecto 0.5 | λ₂ = 1 | Volver a correr. Para una media posterior con λ₂ < 1 use `minnesota_posterior`; no existe posterior NIW con λ₂ < 1. |
| `minnesota_posterior`, `bvar_sv` | — | Siempre | Nada que hacer. |

### ¿Le afecta?

```python
res = minnesota_gibbs(...)   # su llamada en 4.3.0 o antes
res["lambda2"]               # cualquier valor distinto de 1.0 (0.5 por defecto) → le afecta
```

### Qué hay que volver a correr

- **Cualquier análisis construido sobre `minnesota_gibbs` o
  `minnesota_optimal_lambda`.**
- **En este repositorio:** los ejemplos `bvar_fan_chart` (la banda del 80% en
  h = 4 pasa de [-0.876, +0.692] a [-0.947, +0.707]), `posterior_predictive_fan`,
  `mcmc_diagnostics` (la media de la cadena de A1[0,0] pasa de 0.51 a 0.56),
  `gk_robust_from_gibbs` y `glp_lambda_search` (el λ₁ elegido pasa de 0.2 a
  0.3).

---

## 2026-09-30 — intervalos de DiD sintético, y DiD silencioso con resultados en unidades grandes, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** Dos defectos de `did.synthetic_did`, y de
`sdid_multi_cohort` a través de sus estimaciones puntuales.

1. **Intervalos con cobertura insuficiente.** `se`, `lo` y `hi` salían de un
   bootstrap que remuestreaba sólo las unidades donantes y dejaba fijas las
   tratadas, con un intervalo por percentiles, de modo que se omitía el ruido
   propio de las unidades tratadas. Con una unidad tratada, un intervalo nominal
   del 90% cubría el valor verdadero entre el 50% y el 67% de las veces en Monte
   Carlo, y el `se` reportado era cerca de la mitad de la desviación típica
   verdadera (diseño del cuaderno 29: media 0.13-0.14 frente a 0.23-0.24). En
   el ejemplo de la documentación, con 8 unidades tratadas, el `se` era 0.019
   frente a una desviación típica verdadera de cerca de 0.057.
2. **Colapso silencioso a diferencias en diferencias.** Cuando SLSQP reportaba
   que no había tenido éxito, los pesos volvían a ser uniformes (1/N_co para ω,
   1/T_pre para λ) sin advertencia, así que τ̂ era la estimación DiD simple. La
   tolerancia de SLSQP es absoluta, por lo que esto ocurría sobre todo con
   resultados en unidades grandes (los datos del cuaderno 29 multiplicados por
   10 o más, niveles en los cientos). Con los datos de la Proposición 99 de
   California del artículo, puremacro devolvía -27.35, el valor DiD, en lugar
   del SDID -15.6.

El nivel de ruido σ̂ se centraba además periodo a periodo en lugar de como en
la ec. (2.2) del artículo; las estimaciones puntuales se mueven cerca de 1e-2
(cuaderno 29: de -4.0940 a -4.0827).

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `se`, `lo`, `hi` de `synthetic_did` | Siempre | `n_boot=0` (sin inferencia) | Volver a correr. El valor por defecto es ahora el estimador placebo (algoritmo 4) para una o dos unidades tratadas y el bootstrap de unidades (algoritmo 2) en otro caso; reporte el `se` placebo con una o dos unidades tratadas. |
| Estimaciones puntuales de `synthetic_did` y `sdid_multi_cohort` | Falló la resolución de los pesos: pesos uniformes | Los pesos no son uniformes | Volver a correr; el solver corregido no depende de la escala y advierte si no converge. |
| Todas las estimaciones puntuales | Siempre, en cerca de 1e-2, por σ̂ | — | Volver a correr si importa el segundo decimal. |

### ¿Le afecta?

```python
import numpy as np
res = synthetic_did(...)                          # su llamada en 4.3.0 o antes
np.allclose(res.omega.values, 1 / len(res.omega)) # True → el "SDID" reportado era DiD
```

Todo error estándar o intervalo de SDID reportado está afectado.

### Qué hay que volver a correr

- **Todo análisis SDID**: las estimaciones puntuales con resultados en unidades
  grandes y todos los intervalos.
- **En este repositorio:** el cuaderno 29 (inglés y español; su intervalo del
  90% guardado excluye el efecto verdadero), `docs/did.md` y el ejemplo
  `synthetic_did_california_prop99`, que usa datos simulados pese a su nombre.

---

## 2026-09-30 — agregación de Callaway-Sant'Anna, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0; cambia el valor por defecto.**
`did.callaway_santanna` calculaba bien sus efectos por grupo y periodo
ATT(g, t), pero los agregaba con **pesos iguales por cohorte** y presentaba el
resultado como el estimador de Callaway-Sant'Anna. El artículo pondera todo
agregado por el tamaño de la cohorte (arXiv:1803.09015v4: estudio de eventos,
ec. 3.4; resúmenes globales, ecs. 3.10-3.12). `att_overall` era la media simple
de las celdas ATT(g, t) posteriores al tratamiento, que no coincide con ninguno
de los parámetros globales del artículo. La documentación declaraba la regla
sin ponderar pero la atribuía a CS, y el material docente tenía la comparación
al revés: el propio `sun_abraham` de puremacro devolvía exactamente las ecs.
3.4 y 3.10 de CS.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `att_event_study` | Dos o más cohortes identifican un tiempo de evento y difieren en tamaño | Todas las cohortes de ese tiempo de evento tienen el mismo tamaño, o el mismo ATT(g, g+e) | Volver a correr. Ejemplo sin ruido: **2.0 frente a 2.6** en e = 0 con cohortes de 10 y 40 unidades. |
| `att_overall`, y el `overall_att` del diccionario heredado | Cualquier heterogeneidad en ATT(g, t) | Todos los ATT(g, t) son iguales (entonces coinciden todos los resúmenes globales) | Ahora es θ^O_sel (ec. 3.11); `aggregation=` elige los demás. Cuaderno 10: de 1.1292 a 1.1379; curso T05_C: de 7.890 a 6.391 (ahí `aggregation="simple"` conserva 7.890 y es el estimando de media por celda). |

### ¿Le afecta?

Con la versión corregida, `callaway_santanna(..., aggregation="unweighted")`
reproduce exactamente los números antiguos. Córralo junto al valor por
defecto: donde ambos difieran, su resultado antiguo no era el estimando de
Callaway-Sant'Anna.

### Qué hay que volver a correr

- **Cualquier estudio de eventos o ATT global de Callaway-Sant'Anna** sobre un
  panel cuyas cohortes difieren en tamaño y cuyos efectos difieren entre
  cohortes o en el tiempo.
- **En este repositorio:** los cuadernos 10 y 15, las lecciones T05_C y T05_D
  del curso, `docs/did.md`, el ejemplo `did_callaway_santanna_demo` (de 1.0142
  a 1.0278) y el recorrido de 2.0 `00_whats_new_in_puremacro_2_0` (en inglés y
  en español; ATT global de 1.1876 a 1.0904; su estudio de eventos no cambia
  porque sus dos cohortes tienen el mismo tamaño).

---

## 2026-09-30 — errores estándar de Sun-Abraham, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** `did.sun_abraham` agregaba los errores
estándar por cohorte como si las cohortes fueran independientes,
`sqrt(Σ w_g² se_g²)`. La comparación 2×2 de cada cohorte resta las mismas
unidades de control, así que las estimaciones por cohorte están
correlacionadas. Desde 1.10.0 las bandas `lo`/`hi` se construyen con este `se`
(antes de 1.10.0 tenían el error distinto descrito en la entrada del
2026-09-03), así que lo heredaban. El error va en cualquier dirección: con
errores autocorrelacionados o con tendencia, el `se` era 0.74 veces la
desviación típica de Monte Carlo en nuestras corridas (0.62 en la
verificación), y las bandas del 90% cubrían el 77%; con errores iid en
cohortes adyacentes era hasta 1.40 veces demasiado grande. Las estimaciones
puntuales se sostienen.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `se` del estudio de eventos (0.92.0 a 4.3.0), `lo`/`hi` (1.10.0 a 4.3.0) | Dos o más cohortes contribuyen a un tiempo de evento | Una cohorte por tiempo de evento | Volver a correr; el `se` es ahora la desviación típica bootstrap del agregado sobre extracciones conjuntas por unidad (SA 2021, prop. 6). |
| `att_overall` | No se reportaba error estándar | — | Ahora `att_overall_se`/`_lo`/`_hi`. |
| `honest_did` alimentado con un estudio de eventos de Sun-Abraham | Como en la primera fila | Como en la primera fila | Volver a correr; pase `event_study_vcov` a `honest_did(sigma=)`. |

### ¿Le afecta?

Cualquier tiempo de evento con más de una cohorte. En un panel con tendencia
por unidad, el `se` en e = 0 pasa de 0.111 a 0.174 (forma cerrada 0.170).

### Qué hay que volver a correr

- **Cualquier `se` o banda del estudio de eventos de Sun-Abraham** en un tiempo
  de evento con más de una cohorte, y cualquier corrida de `honest_did`
  construida sobre uno.
- **En este repositorio:** los cuadernos 10 y 15 y la lección T05_C del curso.

---

## 2026-09-30 — proyecciones locales con rezagos aumentados, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0; cambia el valor por defecto.** `lp.la_lp` (y
`lp.la_lp_iv`, desde 4.0.0) añadía `extra_lags = max(horizons)` rezagos en
todos los horizontes y daba a los controles sólo `n_lags` rezagos; el método se
atribuía a Plagborg-Møller y Wolf (2021), con una regla "p_aug = p + h" sin
fuente. La proyección local con rezagos aumentados de Montiel Olea y
Plagborg-Møller (2021, *Econometrica* 89(4)) añade un solo rezago. El antiguo
valor por defecto sigue siendo válido asintóticamente, pero su cobertura en
muestras finitas es peor y sus bandas más anchas: con T = 228 cubría entre 0.79
y 0.84 frente a 0.84-0.87 con un rezago extra (nominal 0.90). Además, la
estimación en un h dado dependía del mayor horizonte pedido, y
`horizons=[0]` no tenía aumento.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Estimaciones puntuales y bandas HC0 de `la_lp`, `la_lp_iv` | `extra_lags` no se pasó, o cualquier llamada con `controls=` | Un `extra_lags` explícito, sin controles | Volver a correr. PL de impuestos de Romer-Romer (T = 228, H = 20): la respuesta a 2 años pasa de -3.15 a -2.86, y el pico se mueve de -3.15 en h = 8 a -2.87 en h = 10. |

### ¿Le afecta?

```python
res = la_lp(df, y, x, horizons=H, n_lags=p)   # su llamada en 4.3.0 o antes
res["p_aug"].iloc[0] > p + 1                  # True → el aumento antiguo por defecto
```

Cualquier llamada con `controls=` también está afectada. Para reproducir
exactamente un número antiguo con la versión corregida, pase
`extra_lags=max(horizons), control_lags=n_lags`.

### Qué hay que volver a correr

- **Cualquier resultado de `la_lp` / `la_lp_iv`** obtenido con el `extra_lags`
  por defecto o con `controls`.
- **En este repositorio:** los cuadernos 14 y 50, las lecciones T04_A y T05_B
  de `curso/`, la lección 06 de `notebooks/course`, `docs/lp.md` y el ejemplo
  `la_lp_pmw_demo` (la mediana de la curva de especificación del cuaderno 14
  pasa de -2.30 a -2.64).

---

## 2026-09-30 — valores críticos de b fijo, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** La tabla de b fijo con núcleo de Bartlett
escrita en el código de `inference.llsw_critical_value` y de la clave
`llsw_cv_90` de `inference.hac_fixed_b` no coincidía con Kiefer y Vogelsang
(2005, cuadro I). Su comentario citaba el cuadro 2 de Lazarus, Lewis, Stock y
Watson (2018), que contiene constantes de ancho de banda, no valores críticos,
y sus valores bilaterales al 1% no venían de ninguna tabla publicada. A partir
de b = 0.1, aproximadamente, los valores eran demasiado pequeños, y el déficit
crece con b. El código antiguo interpolaba linealmente entre sus filas
erróneas de b = 0.1 y b = 0.2, así que todo b intermedio está afectado: el
valor bilateral al 5% quedaba un 1% por debajo del correcto con b = 0.10, un 3%
con b = 0.12, un 5% con b = 0.15 (2.265 en lugar de 2.386), un 9% con b = 0.2 y
entre un 14% y un 20% desde b = 0.3 (3.96 en lugar de 4.771 con b = 1). Por eso
las pruebas rechazaban de más y los intervalos eran demasiado estrechos. Los
valores bilaterales al 1% eran demasiado pequeños para todo b desde 0.05. Con
b por debajo de 0.1 al nivel del 10%, y hasta b = 0.08 al nivel del 5%, los
valores eran demasiado grandes, hasta un 8% con b = 0.02, así que esas pruebas
eran conservadoras (una prueba al 10% nominal rechazaba el 8.6% de las veces
con b = 0.05).

| b | Rechazo al 5% nominal | Rechazo al 10% nominal | Rechazo al 1% nominal |
|---|---|---|---|
| 0.05 | 4.5% | 8.6% | 1.1% |
| 0.10 | 5.4% | 9.9% | 1.5% |
| 0.12 | 5.7% | 10.3% | 1.6% |
| 0.15 | 6.2% | 11.0% | 1.8% |
| 0.2 | 7.0% | 12.1% | 2.1% |
| 0.5 | 9.3% | 15.4% | 2.9% |
| 1 | 8.8% | 16.3% | 1.8% |

(tasas de rechazo bajo la nula del estadístico t de `hac_fixed_b` frente al
antiguo `llsw_critical_value(b, alpha)`, datos iid N(0,1), T = 200, promediadas
sobre 200,000 extracciones de la estimación de la varianza. Con los valores
corregidos las mismas pruebas rechazan el 4.8-5.0%, el 9.5-10.1% y el 1.0%.)

Lazarus, Lewis, Stock y Watson recomiendan el ancho de banda de Newey-West
S = 1.3√T, es decir b = 1.3/√T. Ese b está entre 0.1 y 0.2 para T entre
aproximadamente 42 y 169.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `llsw_critical_value`, `hac_fixed_b(...)["llsw_cv_90"]` | b mayor que 0.1 a cualquier nivel (al 5%, desde b = 0.1), con un déficit que crece de aproximadamente un 1% con b = 0.1 a un 9% con b = 0.2 (bilateral al 5%) y más por encima; alpha = 0.01 desde b = 0.05; con b por debajo de 0.1 al 10% y hasta b = 0.08 al 5%, valores demasiado grandes (conservadores) | El b = 0.10 por defecto con alpha = 0.10: el antiguo 1.86 frente a 1.861. `hac_fixed_b` busca ahora su valor en b_eff = (floor(bT) + 1)/T, que da 1.871 con T = 200 y 1.902 con T = 50; ahí la antigua prueba al 10% rechazaba el 9.9% y el 10.4% de las veces. | Volver a correr la prueba o el intervalo. Valores bilaterales antiguos y nuevos con b = 1: 3.05/3.96/6.29 pasan a 3.764/4.771/7.083 al 10/5/1%. |
| Errores estándar, estadísticos t y `vcov` de `hac_fixed_b` | — | Siempre, salvo una corrección de redondeo de un rezago para b·T como 0.29·100 | Nada que hacer. |
| Un alpha fuera de {0.2, 0.1, 0.05, 0.02, 0.01}, o un b fuera de (0, 1] | Se sustituía o recortaba en silencio | — | Ahora lanza `ValueError`. |

### ¿Le afecta?

```python
from puremacro.inference import llsw_critical_value
llsw_critical_value(0.15, 0.05)  # 2.265 en 4.3.0 y antes; 2.386 corregido
llsw_critical_value(1.0, 0.05)   # 3.96 en 4.3.0 y antes; 4.771 corregido
```

Si su b era mayor que 0.1, o su nivel el 1%, vuelva a correr. Con b = 0.10 y
el nivel del 5%, la prueba antigua rechazaba el 5.4% de las veces con T = 200 y
el 5.8% con T = 50; decida si ese margen importa para su resultado.

### Qué hay que volver a correr

- **Cualquier prueba o intervalo construido con `llsw_critical_value` o
  `llsw_cv_90`** con b mayor que 0.1, o con alpha = 0.01. Esto incluye la regla
  de LLSW b = 1.3/√T para muestras de aproximadamente 42 a 169 observaciones.
  Ningún cuaderno ni ejemplo de este repositorio los usa.

---

## 2026-09-30 — error estándar AKM de shift-share, versiones 2.4.0 a 4.3.0

**Corregido después de 4.3.0.** El `se_akm` de `bartik.shift_share_iv`, y con
él el `se` por defecto, el estadístico t, el valor-p y el intervalo, no
implementaba Adão, Kolesár y Morales (2019) cuando la regresión tenía controles
a nivel de unidad además del intercepto, o participaciones de exposición que
no suman uno. El código residualizaba los choques sectoriales brutos sobre una
constante ponderada por las participaciones; AKM usan X̂, la regresión del
instrumento parcializado por los controles sobre las participaciones
(observación 5, ecs. 28 y 39). Tampoco había forma de agrupar sectores (ec.
40). El error va en cualquier dirección: con los datos ADH de la viñeta de
ShiftShareSE el error estándar era 0.1816 en lugar de 0.2101 (13.5% demasiado
pequeño); en un caso sintético con un control regional era 0.4117 en lugar de
0.2120.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Error estándar AKM e inferencia de `shift_share_iv` | Controles además del intercepto, o participaciones que no suman uno | Sólo intercepto y participaciones completas (por ejemplo el ejemplo de `docs/spatial.md`) | Volver a correr. Pase `sector_clusters=` cuando los choques puedan estar correlacionados dentro de grupos de sectores (los grupos SIC de 3 dígitos reproducen el 0.2403730 de la viñeta). `akm_shocks="residualized"` da el número antiguo. |

### ¿Le afecta?

```python
import numpy as np
np.allclose(shares.sum(axis=1), 1.0)   # False → le afecta
# también le afecta si pasó controles además del intercepto
```

### Qué hay que volver a correr

- **Cualquier error estándar, prueba o intervalo AKM** de una regresión con
  controles o participaciones incompletas. Ningún cuaderno de este repositorio
  los usa.

---

## 2026-09-30 — diagnósticos MCMC: estadísticos z de Geweke y tamaño muestral efectivo, versiones 0.92.0 a 4.3.0

**Corregido después de 4.3.0.** `mcmc.geweke_z` estimaba la varianza de largo
plazo de cada segmento con una ventana de Bartlett de sólo 5-7 rezagos en los
segmentos por defecto, demasiado corta para salidas MCMC autocorrelacionadas.
Con cadenas de autocorrelación positiva (Metropolis de paseo aleatorio y Gibbs,
las únicas que le pasa `estimate_dsge_bayesian`) la varianza se subestimaba,
así que cadenas estacionarias se declaraban no convergidas mucho más a menudo
que el 5% nominal: con 5,000 extracciones AR(1) estacionarias, el 9% de las
veces con φ = 0.5, el 29-33% con 0.9, el 46-49% con 0.95 y el 76% con 0.99,
frente a 6%, 7%, 9% y 19% ahora (7% con 0.99 y 50,000 extracciones). Eran
falsas alarmas, no aprobados falsos. Con cadenas antitéticas el estimador
antiguo era algo conservador.

`mcmc.effective_sample_size` sumaba pares de autocorrelaciones desde el rezago
1 sin paso monótono. Nunca podía superar n, y devolvía exactamente n siempre
que la suma de las autocorrelaciones de orden 1 y 2 era negativa, el caso
típico de las cadenas antitéticas (ESS verdadero cercano a 3n con φ = -0.5).

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `geweke_z`, `geweke_z_<param>` y `geweke_z_max` de `estimate_dsge_bayesian` (desde 2.1.0), `trace_summary` | Cadenas persistentes: el valor absoluto de z estaba inflado | Cadenas con autocorrelación despreciable | Volver a correr los diagnósticos. `geweke_z(..., spectrum="bartlett")` reproduce los números antiguos. |
| `effective_sample_size`, ESS bulk y tail de NUTS | Cadenas antitéticas; más ruidoso con autocorrelación positiva | — | Volver a correr; el ESS de NUTS cambia unos pocos puntos porcentuales (cuaderno 43: de 87.2 a 90.6). |
| `gelman_rubin` | — | Siempre: es el estadístico clásico, sin cambios | Use el nuevo `split=True`, o `trace_summary()["R_hat_split"]`, para detectar derivas dentro de una cadena. |

### ¿Le afecta?

Un |z| entre 2 y cerca de 5 en una cadena de Metropolis de paseo aleatorio
larga y bien mezclada, reportado por 4.3.0 o antes, era probablemente una
falsa alarma. En el modelo bayesiano AR(1) sintético de las pruebas,
`geweke_z_max` pasa de 1.19 a 0.78.

### Qué hay que volver a correr

- **Diagnósticos de convergencia** de la estimación bayesiana de DSGE y de
  cualquier cadena pasada a `geweke_z` o `effective_sample_size`.
- **En este repositorio:** el ejemplo `mcmc_diagnostics` y el cuadro de ESS del
  cuaderno 43.

---

## 2026-09-30 — las reproducciones de nowcasts en tiempo real usaban añadas posteriores, versiones 4.1.0 a 4.3.0

**Corregido después de 4.3.0.** Con `as_of` anterior a la última añada del
panel, `realtime_nowcast` calculaba su descomposición de noticias
(`forecast_old`, `revision`, `news_table`, `revision_table`), `vintage_history`
y `news_vs_noise_test` frente a añadas publicadas **después** de `as_of`: la
referencia por defecto era la penúltima añada del panel. Cuando `as_of` era
esa añada, la descomposición era idénticamente cero, y `summary()` podía
imprimir "Analytical Identity Error ... (< 1e-10)" con un error cercano a
1e-2. En un ejemplo con cinco añadas mensuales y `as_of` en la segunda, la
descomposición pasa de forecast_old -0.0828 / revision +0.0270 (error 2.7e-2,
referencia la cuarta añada) a +0.1002 / -0.1560 (error 5.6e-17, referencia la
primera).

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `news_decomposition`, `vintage_history`, `news_vs_noise_test` | `as_of` anterior a la última añada del panel | Sin `as_of` | Volver a correr la reproducción. |
| El nowcast puntual, `forecast_sd`, factores y cargas | — | Siempre: ya eran en tiempo real | Nada que hacer. |
| Código que lee `vintage_history["nowcast"]` | Siempre | — | Lea `vintage_history["last_observed_value"]`: la columna siempre contuvo el último dato observado, nunca un nowcast del modelo. |

### ¿Le afecta?

Cualquier reproducción en pseudo tiempo real que pasó `as_of`. Las llamadas
sin `as_of` sólo cambian en las columnas renombradas de `vintage_history`.

### Qué hay que volver a correr

- **Cualquier descomposición de noticias o prueba de noticias frente a ruido en
  pseudo tiempo real** hecha con `as_of`. El cuaderno 59 usa la última añada,
  así que sus números no cambian; sólo cambia el bloque de resumen impreso.

---

## 2026-09-30 — unidades de Gertler-Karadi (2.3.0 a 4.3.0) y signo de las sorpresas con precios de futuros (0.92.0 a 4.3.0)

**Corregido después de 4.3.0.** Dos convenciones que la documentación no
declaraba llevaban a números erróneos.

1. `GertlerKaradiResult.irf` / `to_frame()` siempre ha contenido desviaciones
   **en niveles** x_t - x_ss, porque el modelo se linealiza en niveles.
   Multiplicarlas por 100 y llamar "%" al resultado es incorrecto para toda
   variable cuyo estado estacionario no sea 1. Con la solución por defecto
   (OccBin, choque de calidad del capital de -5%), los valores en el impacto en
   porcentaje son: capital -5.06 (no -28.66; mínimo -13.49, no -76.38),
   patrimonio neto -49.15 (no -67.88), inversión -7.10 (no -1.01), producto
   -1.60 (no -1.35), consumo -0.65 (no -0.35). Q no está afectado porque
   Q_ss = 1.
2. `hfi.gk2015_surprise` devuelve `post - pre`, escalado, y su docstring
   aceptaba un **precio** de futuros o una tasa implícita. Con precios
   (cotizados como 100 - tasa) la sorpresa salía con el signo invertido: un
   endurecimiento de 25 pb daba -0.5 en lugar de +0.5. Eso invierte las
   respuestas de un SVAR con proxy (`proxy_svar` no normaliza el signo por
   defecto) e intercambia los choques monetarios y de información de
   Jarociński-Karadi.

`aggregate_to_period` además suma las sorpresas dentro del mes calendario,
mientras que Gertler y Karadi (2015, nota 11) usan un promedio del periodo; los
resultados que afirman replicar su instrumento mensual deben usar el nuevo
`method="gk2015"`.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Resultados de `solve_gertler_karadi` reportados en porcentaje | Las desviaciones en niveles se multiplicaron por 100 | Desviaciones en niveles reportadas como tales; Q | Use `to_frame(units="pct")`. |
| `gk2015_surprise` y lo construido sobre ella | Se pasaron precios de futuros | Se pasaron tasas implícitas | Volver a correr con `quote="price"`, o pasar tasas implícitas. |

### ¿Le afecta?

```python
res = solve_gertler_karadi()
res.to_frame()["K"].iloc[0]   # -0.2866: desviación en niveles (K_ss = 5.66), es decir -5.06%, no -28.66%
```

Para la sorpresa: si sus datos de entrada eran precios cercanos a 95 y no tasas
cercanas a 5, sus sorpresas tienen el signo equivocado.

### Qué hay que volver a correr

- **Cualquier respuesta de Gertler-Karadi reportada en porcentaje**, y
  **cualquier SVAR con proxy o descomposición de choques de información**
  construida con sorpresas cotizadas en precios.
- **En este repositorio:** la lección T02_F del curso (inglés y español)
  imprimía los números mal escalados, y el ejemplo `hfi_gertler_karadi` usaba
  un instrumento de puro ruido cotizado en precios.

---

## 2026-09-30 — respuestas de Blanchard-Quah de variables en niveles, todas las versiones (comportamiento documentado, ahora seleccionable)

**Cambiado después de 4.3.0; el valor por defecto no cambia.**
`var.identify.bq_svar` acumula la respuesta de todas las variables, lo cual
sólo es correcto cuando todas las columnas de `Y` están diferenciadas, y el
docstring de su resultado llamaba a la salida "respuesta en niveles de cada
variable". En el sistema canónico de Blanchard-Quah (Δlog PNB, tasa de
desempleo en niveles), la respuesta del desempleo salía como una suma
acumulada que converge a una constante distinta de cero en lugar de volver a
cero, y lo mismo sus bandas; en una comprobación con proceso generador
conocido daba +0.687 en h = 40 frente a un valor verdadero de 0.000. El
comportamiento estaba documentado, así que no es un error en el sentido de las
reglas de abajo, pero confundió al material docente.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| Respuestas y bandas de `bq_svar` | Una columna de `Y` en niveles | Todas las columnas diferenciadas | Pase `cumulate=[índices de las columnas diferenciadas]`, por ejemplo `cumulate=[0]`; la elección se aplica a la estimación puntual y a cada extracción bootstrap. |

### ¿Le afecta?

Si alguna variable de su VAR de Blanchard-Quah entró en niveles y leyó su
respuesta de `bq_svar` como respuesta en niveles, le afecta. Diferenciar la
respuesta después (`np.diff`) recupera la estimación puntual pero no las
bandas.

### Qué hay que volver a correr

- **Respuestas de Blanchard-Quah de variables en niveles**, y sus bandas.
- **En este repositorio:** `teaching.bq_canonical.bq_gdp_urate`, la lección
  T02_B de `curso/` (que acumulaba por segunda vez), las lecciones 02 y 05 de
  `notebooks/course`, el ejemplo `gali_1999_hours` y el cuaderno 35 (horas en
  niveles).

---

## 2026-09-30 — Aiyagari continuo: política de los hogares, vaciado del mercado y `converged`, versiones 3.3.0 a 4.3.0

**Corregido después de 4.3.0.** `solve_aiyagari_continuous` detenía en
silencio el EGM de los hogares tras 500 iteraciones, devolvía
`converged=True` incondicionalmente e ignoraba `solver` y los argumentos
desconocidos; sus envoltorios `AiyagariContinuousModel.solve` y
`AiyagariContinuousEquilibrium.solve` usaban por defecto una tolerancia de
Brent `xtol=1e-6`, demasiado laxa para la tolerancia documentada de vaciado de
1e-4. Donde β(1+r*) está cerca de uno, la política devuelta no era el punto
fijo, y el error de vaciado reportado, calculado con esa política, subestimaba
el exceso de demanda verdadero: 1.6e-2 verdadero frente a -2.5e-6 reportado en
el ejemplo documentado, 1.67e-2 frente a 5.95e-4 para
`AiyagariContinuousModel(n_z=3).solve(N_k=150)`, -2.9e-3 frente a 1.4e-9 con
β = 0.995. El error en r* es pequeño en todos los casos comprobados, como mucho
0.0006 puntos porcentuales (ejemplo documentado: de 0.039421 a 0.039416), con
K* moviéndose cerca de 4.5e-4 y el Gini cerca de 5e-4.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `solve_aiyagari_continuous` y los envoltorios | Calibraciones con n_z = 3 y ρ_z = 0.9 (el ejemplo documentado y el estado estacionario del documento de transiciones), β trimestral ≥ 0.99, `a_max` grande, o el antiguo `xtol` de los envoltorios | La calibración por defecto (n_z = 5): r dentro de 1e-9; cuaderno 52 (r a 1e-9, K a 1e-7); cuaderno 67 (r a 1e-10) | Volver a correr; revise `eq.converged`, `eq.metadata["egm_iterations"]` y `eq.metadata["clearing_ok"]`. |
| `solve_aiyagari_continuous` con `solver="collocation"` o `"fem"` | Siempre: el resultado era el de EGM | — | Ahora lanza `ValueError`. |
| `solve_continuous_transition` con un choque permanente | Su estado estacionario terminal tenía el mismo tope y la misma tolerancia | Choques transitorios | Volver a correr las transiciones con choques permanentes; el estado estacionario terminal usa ahora el mismo bucle y las mismas tolerancias. |

### ¿Le afecta?

Vuelva a resolver con la versión corregida: si `eq.metadata["egm_iterations"]`
supera 500, el resultado antiguo se calculó con una política de los hogares
sin converger.

### Qué hay que volver a correr

- **Estados estacionarios calculados con n_z = 3, con β ≥ 0.99, o con los
  envoltorios y su antiguo valor por defecto**, y las transiciones construidas
  sobre ellos.

---

## 2026-09-30 — `steady()` devolvía en silencio estados estacionarios estructuralmente singulares, versiones 2.6.0 a 4.3.0

**Corregido después de 4.3.0.** `dsge.steady(solve_algo="block")` es la ruta
por defecto de `build()`, `build_dynare()` y `load_mod()`. Cuando el sistema
estático no tenía un emparejamiento completo entre ecuaciones y variables pero
una resolución `hybr` del sistema completo convergía, devolvía ese punto sin
advertencia. Ese punto es un miembro arbitrario de un continuo de estados
estacionarios y depende del valor inicial: en `{x-1, 2x-2, x+y}` con `z` en
ninguna ecuación, `z` volvía como el valor inicial más 0.769133 (1.0 daba
1.769133, 7.5 daba 8.269133); un paseo aleatorio `y' = y + eps` con `c = 2y`
devolvía `(1.5, 3.0)` desde `(4, 3)` y `(0, 0)` desde `(1, 0)`.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `steady()`, `build()`, `build_dynare()`, `load_mod()` | Ecuaciones estructuralmente duplicadas (un subconjunto de ecuaciones con menos variables que ecuaciones), o una variable ausente del sistema de estado estacionario | Todo modelo con emparejamiento completo: sus resultados no cambian | Ahora lanzan `StructuralSingularityError`; los niveles reportados los fijaba el valor inicial. |
| Lo mismo, estructura de raíz unitaria (toda ecuación sobredeterminada se reduce a 0 = 0) | Siempre | — | La misma respuesta, ahora con una `StructuralSingularityWarning` que avisa de que el nivel está libre. |

Las ecuaciones numéricamente colineales con incidencia completa (`x + y = 2` y
`2x + 2y = 4`) siguen sin detectarse: la comprobación es estructural.

### ¿Le afecta?

Vuelva a calcular el estado estacionario con la versión corregida. Una
`StructuralSingularityError` o `StructuralSingularityWarning` significa que el
estado estacionario antiguo era un punto arbitrario de un continuo. Para
recuperar el punto antiguo, con el aviso, pase `allow_singular=True` a
`steady()`, `build()`, `build_dynare()` o `load_mod()`, o envuelva la llamada en
`with allow_structural_singularity():`
(`from puremacro.dsge.steady import allow_structural_singularity`). Un modelo
construido guarda el informe del solver en `model.steady_state_info`.

### Qué hay que volver a correr

- **Cualquier modelo que produjo un estado estacionario y tiene alguna de las
  estructuras anteriores**, y todo lo calculado a partir de él.

---

## 2026-09-30 — solvers de política óptima: sesgo cero silencioso, pérdidas de relleno y un parámetro inerte, versiones 2.6.0 a 4.3.0

**Corregido después de 4.3.0.**

1. **Sesgo cero silencioso** (3.1.0 a 4.3.0).
   `DiscretionaryPolicyResult.stabilization_bias` valía exactamente 0.0, sin
   advertencia, siempre que la resolución interna de `lq_commitment` fallaba o
   la pérdida de compromiso elegida era NaN, y `summary()` omitía la línea. Un
   sesgo reportado de exactamente 0.0 puede ser por tanto una comparación
   fallida. Los valores en calibraciones bien planteadas no cambian
   (1.513166984879534 en la base del cuaderno 66).
2. **Pérdida incondicional sin sentido.** En un modelo cuya transición en lazo
   cerrado tiene una raíz unitaria, la `loss` de compromiso podía ser un número
   finito sin significado (4.9e15 con un choque de costos con raíz unitaria).
   Ahora es NaN.
3. **Valores de relleno en `osr`.** `loss_initial` tomaba el valor del
   argumento `penalty` (1e6) cuando fallaban los momentos de referencia, con
   sólo una UserWarning; `loss_opt` era en silencio el último objetivo del
   optimizador (quizá una penalización de 1e8) cuando el modelo no podía
   volver a resolverse con los coeficientes devueltos; y
   `variance_reduction_pct` valía 0.0 donde faltaba una varianza, de modo que
   `summary()` imprimía "Loss reduction: 0.00%". Los tres son ahora NaN con una
   `RuntimeWarning`.
4. **El parámetro `timeless` no hacía nada.** `lq_commitment` y `ramsey_model`
   devolvían siempre la ley de movimiento común al plan de Ramsey y a la regla
   intemporal, respuestas y `conditional_loss` desde el estado estacionario,
   donde ambos coinciden, y `loss` promediada sobre la distribución
   estacionaria. El código que pasaba `timeless=False` esperando otra
   condición inicial obtenía el mismo resultado.
5. **Precisión de `osr`.** Con las tolerancias de Nelder-Mead por defecto de
   SciPy, `osr` encontraba la asignación de la regla óptima sólo hasta cerca de
   1e-5 cuando el óptimo estaba en una cota (hasta 2.7e-5 en calibraciones
   aleatorias de Clarida-Galí-Gertler). Las pérdidas eran exactas a 2.5e-10, y
   los coeficientes reportados con cuatro decimales no están afectados.
6. **Cita.** La documentación y los docstrings atribuían a Jensen y McCallum
   (2002) la inversión con β bajo de la ordenación incondicional; es de Sauer
   (2010, prop. 2).

### ¿Le afecta?

```python
res.stabilization_bias == 0.0   # en 4.3.0: quizá una comparación fallida; vuelva a correr
osr_res.loss_initial == 1e6     # el valor de relleno de penalty, no una pérdida
```

Cualquier `variance_reduction_pct` de exactamente 0.0, y cualquier código que
dependiera de `timeless=False`, están afectados como se describe.

### Qué hay que volver a correr

- **Sesgos de estabilización de exactamente 0.0, resultados de `osr` con
  pérdidas de relleno, y asignaciones de `osr` citadas con más de cinco cifras
  significativas.**
- **En este repositorio:** los números del cuaderno 66 no cambian; sólo cambia
  su cita de la inversión con β bajo.

---

## 2026-09-30 — el "estado estacionario estocástico" de una solución podada es la media ergódica, versiones 2.0.0 a 4.3.0

**Aclarado después de 4.3.0; no cambia ningún número.**
`stochastic_steady_state()` de `PrunedDSGESolution` (desde 2.0.0) y
`Order3PrunedSolution` (desde 2.8.0) devuelve E[x] y E[y] de la solución
podada. Su nombre y su docstring ("estado estacionario ajustado por riesgo")
sugerían el estado estacionario con riesgo sin choques. La media ergódica es
el término de riesgo precautorio ½ g_σσ σ² más curvatura por dispersión
(½ g_xx vec Ω + ½ g_uu vec σ²Σ_u), un efecto de Jensen presente incluso cuando
g_σσ = 0. Los números siempre fueron correctos como medias ergódicas.

Leerlos como el estado estacionario con riesgo, o como desplazamientos
precautorios, es incorrecto. En el modelo RBC de
`tests/fixtures/dynare_live/rbc.mod`, la media ergódica del capital es +0.0785%
de su estado estacionario, mientras que el estado estacionario con riesgo y el
término de riesgo son -0.0045% (el propio ghs2 de Dynare para el capital es
-9.76e-5). En el recorrido de la 2.0 sólo 1.79e-5 del desplazamiento de
6.77e-4 del log del capital es el término de riesgo.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `stochastic_steady_state()` | Su salida descrita como estado estacionario con riesgo o como efecto precautorio | Descrita como la media ergódica (incondicional) | Use `ergodic_mean()` para la media, `risky_steady_state()` para el punto fijo sin choques y `risk_decomposition()` para ver qué término mueve el desplazamiento. |

### ¿Le afecta?

Sólo si interpretó la salida como el estado estacionario con riesgo o como un
efecto precautorio; `risk_decomposition()` en la versión corregida muestra qué
parte del desplazamiento es riesgo y qué parte curvatura.

### Qué hay que volver a correr

- **Nada numérico.** Reformule cualquier afirmación "precautoria" basada en él.
- **En este repositorio:** el cuaderno 68 §4, la lección T02_B del curso
  ("Precautionary Premium" del capital +0.07711%, cuyo estado estacionario con
  riesgo es -0.02807%), el recorrido de la 2.0 §7 y `docs/dsge_build.md` usan
  esa etiqueta.

---

## 2026-09-30 — diagnósticos del solver comercial y cierre de la política hicksiana, versiones 4.2.0 y 4.3.0

**Corregido después de 4.3.0.** Diagnósticos que describían un objeto distinto
del devuelto, y un cierre que las búsquedas de política no podían alcanzar.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `trade.solver.solve_cyprus_manifold_step` (alias `solve_cyprus_manifold`, 4.3.0) | `max(abs(dw)) > max_disp`, así que el paso se recorta: `(residual, converged)` describían el paso sin recortar. El cuaderno 64 imprimía 4.4e-16 y `converged=True` para un paso cuyo residuo es 3.208, todo el lado derecho. El `idx_cyp=15` por defecto es Colombia, no Chipre (17) | Pasos sin recortar, con `idx_cyp` pasado explícitamente | La versión corregida reporta el residuo del paso devuelto; `return_info=True` da los valores sin recortar. Sin `eval_cyp_fn` el paso es una resolución lineal exacta reescalada uniformemente: un paso acotado, no una solución estabilizada. |
| Docstrings de `solve_keller_pac` y `metadata["fold_points"]` (4.3.0) | Los docstrings afirmaban un pliegue en σ_fold ≈ 0.1238 donde Newton diverge; PAC no registra ningún pliegue ahí y Newton converge en 8 iteraciones. `fold_points` listaba cada paso con `tau_lambda <= 0`, una entrada por cada paso restante para un solo estancamiento de precios de factores | — | No cite PAC como recorrido de un pliegue CGE. Lea los eventos fusionados y su `kind`: `factor_price_boundary` es una esquina donde un precio de factor llega a cero, no un pliegue. |
| `compute_unilateral_optimal_tariff`, `solve_multilateral_nash_tariffs`, `compute_welfare_payoff_matrix` hicksianos (4.3.0) | Saldos exteriores de referencia distintos de cero: sólo se alcanzaba el cierre del numerario, así que los pagos dependían de qué país se listaba primero (VE de A por un arancel del 10% en la tabla del cuaderno 63: 2.6308% o 2.4952%) | Saldos exteriores nulos | Pase `foreign_saving_units="world_income"` y una referencia resuelta con él: 2.5404% en cualquier orden. |
| Etiquetas de frontera (`metadata["boundary"]`, `["best_response_boundaries"]`, 4.3.0) | `tariff_max <= tol`: las respuestas en el techo se etiquetaban `"lower"` | Techos por encima de `2*tol` | Corregido: se usa la cota más cercana. |
| `compute_stone_geary_final_demand` (4.2.0 y 4.3.0) | Gasto de subsistencia por encima del presupuesto: la canasta `c_bar` gastaba de más en silencio (169% con μ = 0.9 y 5% del ingreso de referencia en una tabla 2x2) | Σ_s μ_s θ_s0 por debajo de tanh(3)/3 ≈ 0.332 | Ahora se emite una `RuntimeWarning`; revise el presupuesto. |

### ¿Le afecta?

Para el paso de Chipre: `max(abs(dw)) == max_disp` en un paso devuelto
significa que se recortó y que el residuo reportado no era el suyo. Para las
búsquedas hicksianas: vuelva a correr con los países en el orden inverso; si
cambia un pago, el cierre dependía del orden.

### Qué hay que volver a correr

- **Cualquier paso de Chipre recortado reportado como convergido**, y cualquier
  afirmación de pliegue basada en `fold_points`.
- **Búsquedas arancelarias hicksianas** sobre tablas con saldos exteriores
  distintos de cero.
- **En este repositorio:** los cuadernos 63 y 64 (inglés y español).

---

## 2026-09-23 — momentos teóricos de tercer orden, versiones 2.8.0 a 4.3.0

**Corregido después de 4.3.0** (véase la sección Unreleased de `CHANGELOG.md`).
`Order3PrunedSolution.theoretical_moments()`, y el `theoretical_moments` que
devuelve `stoch_simul(order=3)`, reportaban la covarianza, las correlaciones y
las autocorrelaciones **de primer orden** de una solución de tercer orden. Las
columnas Skewness y Kurtosis valían 0 y 3 para todas las variables. El docstring
decía que los momentos coincidían con el `stoch_simul` de Dynare en orden 3, y
`docs/es/dsge_higher_order.md` describía una asimetría y una curtosis exactas en
forma cerrada. La columna Mean era correcta: incluye la corrección por riesgo de
segundo orden, que es la media exacta de la solución podada de tercer orden con
choques gaussianos.

El error es de orden sigma^4 frente a una varianza de orden sigma^2, así que es
pequeño cuando los choques son pequeños y dominan los términos de primer orden.
En el modelo RBC de `tests/fixtures/dynare_live`, las varianzas del producto, el
consumo y el capital son 0.15%, 0.10% y 0.17% mayores que las de primer orden.
Una variable sin respuesta de primer orden se reportaba con varianza cero y
autocorrelaciones indefinidas. En `correlated_cubic.mod`, `y` tiene varianza
0.788 y autocorrelación de orden 1 de 0.475; en `quadratic_feedback.mod`, 0.0283
y 0.742.

| Superficie | Condición afectada | No afectada cuando | Qué hacer |
|---|---|---|---|
| `theoretical_moments()` de una solución de orden 3: covarianza, correlación, autocorrelaciones (2.8.0 a 4.3.0) | Cualquier coeficiente de segundo o tercer orden distinto de cero | La columna Mean; una solución con todos los coeficientes de orden superior nulos | Volver a correr con la versión corregida, que devuelve los momentos exactos del espacio de estados podado. |
| Sus columnas Skewness y Kurtosis | Siempre: eran valores de relleno | Nunca | La versión corregida reporta NaN. Use `ergodic_moments()`, que estima ambas con una simulación podada. |
| `stoch_simul(order=3)` | Su bloque `theoretical_moments`, como arriba | Momentos simulados (`periods > 0`), respuestas al impulso y reglas de decisión | Como arriba. |

`ergodic_moments()` lanzaba `KeyError: 'Std.Dev'` en las mismas versiones, así
que no devolvía números. Los momentos calculados con `simulate()` no se ven
afectados.

### Qué volver a correr

- **Cualquier varianza, desviación típica, correlación o autocorrelación**
  reportada con `theoretical_moments()` de una solución de tercer orden,
  incluidas las tablas de `stoch_simul(order=3)`.
- **Cualquier asimetría o curtosis** tomada de esas tablas: eran valores de
  relleno, no estimaciones.

La versión corregida coincide con precisión de máquina con las medias y
varianzas podadas de Dynare 8 en seis modelos. No coincide con las
autocorrelaciones de orden 3 de Dynare, que omiten un término de correlación;
una forma cerrada y las propias simulaciones largas de Dynare coinciden con
puremacro (véase `docs/es/dsge_higher_order.md` §2.3).

---

## 2026-09-23 — equilibrio comercial flexible, versiones 4.2.0 y 4.3.0

**Corregido después de 4.3.0** (véase la sección Unreleased de `CHANGELOG.md`).
`solve_flexible_trade_equilibrium` devolvía el equilibrio heredado
Cobb-Douglas/Leontief, etiquetado como resultado flexible, para toda
configuración de CES anidada, Stone-Geary (LES), márgenes variables y
capacidad. La configuración se guardaba en `metadata["config"]` pero nunca
entraba en las ecuaciones de equilibrio: el despacho por defecto (Newton) la
recibía sin leerla; solo `method="sparse_lu"` o la ruta `"condensed"` heredada la
usaban, para el patrón de dispersión del jacobiano. La solución convergía, no
emitía ningún aviso e informaba la configuración solicitada.

En una calibración sintética de 3 países y 3 sectores con un arancel del 25%,
las versiones v4.2.0 y v4.3.0 devuelven un vector solución idéntico bit a bit al
de la configuración por defecto (diferencia absoluta máxima 0.0, mismas
iteraciones y mismo residuo) para `rho_va=0.7, sigma_y=0.5`,
`subsistence_ratio=0.2`, `variable_markups=True` y
`capacity_margins={(0, 0): 0.3}`. El cuaderno 62 ejecutado que acompañaba a esas
versiones informa sus soluciones homotética, Stone-Geary y totalmente flexible
(CES, subsistencia y márgenes de oligopolio) sobre la tabla 77x11 incluida con
la misma norma final del residuo (2.7575e-02); sus cuadros de bienestar y de
márgenes solo difieren por el posprocesamiento de un mismo equilibrio.

| Superficie | Condición afectada | No afectado cuando | Qué hacer |
|---|---|---|---|
| `solve_flexible_trade_equilibrium` (4.2.0, 4.3.0), `method` por defecto | Cualquier ajuste tecnológico distinto del valor por defecto (`rho_va`, `sigma_va`, `sigma_y`, `sigma_inter`, `capacity_margins`), subsistencia Stone-Geary (`subsistence_shares`, `subsistence_ratio`, `mu_s`) o `variable_markups=True` | Todos los ajustes flexibles están en su valor por defecto: el modelo flexible coincide entonces con el heredado y el resultado es correcto | Vuelva a correrlo con la versión corregida y compruebe `result.metadata["flexible_settings_applied"]`. |
| La misma función con `method="quasi_condensed"` explícito (4.2.0, 4.3.0) | Los mismos ajustes | Como arriba | Esa ruta evaluaba algunos ajustes de tecnología, capacidad y márgenes pero ignoraba la subsistencia Stone-Geary (`subsistence_ratio=0.2` devolvía la misma solución); trate sus resultados como afectados y vuelva a correrlos con la versión corregida. |
| Cifras de bienestar, VE y márgenes calculadas a partir de esos resultados | Cualquier ejecución afectada | Como arriba | Se derivaron del equilibrio heredado; vuelva a correrlas. |

Las ejecuciones con la configuración flexible por defecto no se ven afectadas,
con una excepción: `solve_trade_equilibrium(method="quasi_condensed")` con
`sigma_y > 0` o con una `config` flexible resuelve ahora el modelo flexible
corregido, y sus resultados cambian. La versión corregida cambia además el
abastecimiento por defecto de la demanda final del solucionador flexible a
coeficientes fijos (el abastecimiento de Armington requiere un `sigma_trade`
explícito).

En la versión corregida, los modelos con 100 celdas país-sector o menos
resuelven las ecuaciones flexibles por la ruta de Newton cuasi-condensada. Por
encima de ese tamaño (incluida la tabla 77x11 incluida en el paquete), o con un
`method=` explícito distinto de `"quasi_condensed"`, una llamada con ajustes
flexibles activos lanza `ValueError` en lugar de ignorarlos. Pase
`method="quasi_condensed"` para resolver las ecuaciones flexibles con un
jacobiano denso, o `allow_legacy_fallback=True` para obtener el equilibrio
heredado con un `RuntimeWarning` y `metadata["flexible_settings_applied"] = False`.

### Qué hay que volver a correr

- **Cualquier contrafactual, cifra de bienestar o de márgenes** publicada a
  partir de `solve_flexible_trade_equilibrium` en 4.2.0 o 4.3.0 con un ajuste
  distinto del valor por defecto.
- **El cuaderno 62 tal como se ejecutó en 4.2.0 y 4.3.0**: el barrido de
  `rho_va`, la comparación homotética frente a Stone-Geary y la comparación
  competencia frente a oligopolio muestran todos el equilibrio heredado.
- **El cuaderno 62 de este repositorio se reescribió y se volvió a ejecutar**
  (en inglés y en español): evalúa los bloques flexibles de costes, demanda y
  precios a precios dados y resuelve un equilibrio de referencia pequeño con el
  solucionador auditado de contabilidad consistente; ya no informa equilibrios
  flexibles sobre la tabla incluida.

---

## 2026-09-23 — curvatura de la colocación ignorada o mezclada, versiones 3.3.0 a 4.3.0

**Corregido después de 4.3.0** (véase la sección Unreleased de `CHANGELOG.md`).
`CollocationProblem(method="bellman")` resolvía el modelo de crecimiento con
utilidad logarítmica fuera cual fuera `params["sigma"]` o `params["gamma"]`.
`CollocationProblem` con `params={"gamma": g}` (sin la clave `"sigma"`)
resolvía la política de utilidad logarítmica, y 4.3.0 además valoraba esa
política logarítmica con CRRA(g) en `value_coefficients`/`value()`. Las
sensibilidades TFI calculadas sobre ese problema (`compute_ift_gradients`,
`equilibrium_parameter_jacobian`, `gmm_objective_and_gradient`) eran las del
problema logarítmico, con una columna de `gamma` nula.

| Superficie | Condición afectada | No afectado cuando | Qué hacer |
|---|---|---|---|
| `CollocationProblem(method="bellman")` | `sigma` o `gamma` distinto de 1 | Utilidad logarítmica (`sigma` o `gamma` igual a 1 o ausente) | Vuelva a resolver con la versión corregida. |
| Políticas y valores auxiliares de `CollocationProblem` | Curvatura escrita como `params={"gamma": g}` sin la clave `"sigma"`, `g != 1` | `method="euler"` con la curvatura escrita como `"sigma"` | Vuelva a resolver y a calcular los valores derivados. |
| Gradientes TFI sobre esos problemas | Cualquier resolución afectada, y la columna de `gamma` en todos los casos | Utilidad logarítmica respecto de parámetros distintos de `gamma` | Vuelva a calcular los gradientes; la columna de `gamma` era idénticamente nula. |

En la versión corregida una sola curvatura CRRA (`sigma`, si no el alias
`gamma`, si no 1) entra en el residuo de Euler, el arranque de Coleman, el
objetivo de Bellman y los coeficientes de valor auxiliares; la utilidad
logarítmica es idéntica bit a bit a 4.3.0.

---

## 2026-09-23 — entradas comerciales leídas como otra tarifa u otra disposición, versión 4.3.0

**Corregido después de 4.3.0** (véase la sección Unreleased de `CHANGELOG.md`).
Cuatro funciones comerciales que aparecieron en 4.3.0 calculaban su resultado
para una entrada distinta de la recibida, sin excepción ni advertencia: un
objetivo de continuación, una tarifa de viabilidad, las etiquetas de los usos
finales de un archivo FIGARO y el suelo del valor añadido de la regularización
MRIO.

| Superficie | Condición afectada | No afectado cuando | Qué hacer |
|---|---|---|---|
| `solve_keller_pac` (4.3.0) | `tau_target` pasado como un escalar de NumPy distinto de `np.float64` (`np.int64`, por ejemplo de `np.arange`; `np.float32`; `np.bool_`) o como un arreglo 0-d: solo se gravaban las importaciones intermedias, mientras que un número de Python grava también las importaciones de demanda final | `tau_target` es un `int`, `float` o `bool` de Python, un `np.float64` o una tarifa en arreglo con al menos una dimensión | Vuelva a resolver con la versión corregida, o pase `float(tau_target)`. |
| `check_hawkins_simon_viability` (4.3.0) | `tau` pasado como lista de Python (evaluada como la línea base sin aranceles, fueran cuales fueran sus valores), como arreglo 0-d (un multiplicador de toda la matriz de costes) o como arreglo de forma `(M, M)`, `(M, ns, nc)` o `(ns, nc, ns, nc)` cuyas entradas diagonales (mismo país y sector) no son 1, que una heurística podía leer como tasas donde `solve_trade_equilibrium` lee multiplicadores (`np.ones((M, ns, nc)) * 1.10` pasaba a ser un multiplicador de 2.10) | `tau` es `None`, un número escalar de Python o de NumPy, un vector de tasas `(nc,)` de NumPy o un arreglo de multiplicadores cuyas entradas diagonales son 1 | Vuelva a hacer la comprobación con la versión corregida; un veredicto "viable" para esa entrada puede ser erróneo. |
| `load_figaro` sobre un archivo armonizado (4.3.0) | El archivo presenta los usos finales de cada país en un orden distinto de `P3_S14, P3_S15, P3_S13, P51G, P5M`, por ejemplo el orden de Eurostat `P3_S13, P3_S14, P3_S15, P51G, P5M`: las columnas de uso final se etiquetaban por posición, de modo que el consumo del gobierno y el de los hogares intercambiaban etiquetas | El archivo sigue el orden de puremacro, o los datos son sintéticos (`fallback_to_synthetic=True`) | Vuelva a cargar con la versión corregida, que lee los usos finales por etiqueta. |
| `regularize_mrio_table`, y `package_mrio_to_calibration_result` con los cargadores que la llaman (`load_figaro`, `load_exiobase`, `load_wiod`, `load_eora`, `load_oecd_icio_granular`) con `regularize=True` (4.3.0) | Un nodo cuyo valor añadido se lleva al suelo tiene una producción inferior a 2 en las unidades de la tabla (2 M USD para los cargadores): el suelo `max(1e-3 Y, 1)` fijaba su valor añadido en 1, por encima de la producción cuando esta es inferior a 1, y cargaba la diferencia a sus impuestos netos de subvenciones | Todo nodo llevado al suelo tiene una producción de al menos 2, o `regularize=False` | Vuelva a calibrar con la versión corregida, cuyo suelo es `max(1e-3 Y, min(1, 0.5 Y))` y que registra cada nodo llevado al suelo. |

Medido sobre la etiqueta de la versión 4.3.0 y sobre la versión corregida:

- `solve_keller_pac` sobre una tabla de prueba de 2 países y 2 sectores:
  `tau_target=np.int64(1)` devolvía `x_sol[:4]` = [0.0965, -0.0050, 0.1579,
  0.1302] con `converged=True`; `tau_target=1.0` devuelve [0.2008, 0.0809,
  0.3446, 0.3215], lo mismo que la versión corregida devuelve para
  `np.int64(1)`.
- `check_hawkins_simon_viability` sobre la misma tabla: `[0.5, 0.5]` devolvía
  rho 0.3379685 (la línea base) en lugar de 0.3762996; `np.array(0.1)`
  devolvía 0.0337969 en lugar de 0.3456204; `np.ones((M, ns, nc)) * 1.10`
  devolvía 0.7097339, el radio para un multiplicador de 2.10, en lugar de
  0.3717654.
- `load_figaro` sobre un archivo de 2 países y 2 sectores en el orden de
  Eurostat: las participaciones de uso final (`theta`) diferían de las de los
  mismos datos en el orden de puremacro hasta en 0.23, y un equilibrio con un
  arancel del 10% hasta en 9.7e-4 (mayor incógnita 12.8). La versión corregida
  da calibraciones idénticas para ambos órdenes.
- `load_oecd_icio_granular` sobre el archivo ICIO 2019 de la OCDE
  (`2019_SML.csv`): 4.3.0 elevaba 41 entradas de valor añadido en 9.84 M USD
  en total y dejaba 10 nodos con valor añadido por encima de la producción
  (impuestos netos de subvenciones de hasta -467% de la producción); la
  versión corregida eleva 36 entradas en 2.74 M USD, ninguna por encima de la
  producción.

### Qué hay que volver a correr

- **Trayectorias de continuación** de `solve_keller_pac` cuyo objetivo venía de
  un arreglo o de una variable de bucle de NumPy (por ejemplo
  `for rate in np.arange(1, 4)`).
- **Comprobaciones de viabilidad** de tarifas en lista, arreglos 0-d o
  arreglos de multiplicadores con entradas diagonales distintas de 1.
- **Calibraciones de archivos FIGARO armonizados** cuyas columnas de uso final
  no seguían el orden de puremacro, incluido cualquier resultado por
  categoría.
- **Calibraciones de tablas reales con nodos pequeños** construidas con los
  cargadores de 4.3.0, y todo contrafactual resuelto sobre ellas.

---

## 2026-09-22 — procedencia de la tabla OCDE 77x11 incluida en el paquete, todas las versiones

**No es un defecto de código; es un hallazgo de procedencia de datos.
Documentado y protegido después de 4.3.0.**

`puremacro/trade/_datafiles/icio_77c_11s.npz` es una copia bit a bit del arreglo
MATLAB `data_77c_11s.mat` que la cadena original de cálculo de mala asignación
sectorial construyó a partir de una exportación `data_2020_SML.csv` de las tablas
ICIO de la OCDE. Esa exportación está dañada: los valores con tres o cuatro
decimales perdieron el punto decimal y los valores por debajo de 0.001 pasaron a
cero (su MD5 `d1b887aaafa54ab3f28fde78fcd21cdf` figura como corrupto en el registro
del espacio de investigación que produjo los resultados revisados del artículo).

Medido frente a la publicación limpia OCDE 2020 (MD5 `d3e0f4979d85d6c0bb7cf4c43e324287`):

| Magnitud | Tabla incluida / exportación dañada | Publicación OCDE 2020 limpia |
|---|---:|---:|
| Valor agregado mundial, agregación 77x11 (millones de USD) | 7.05e11 | 7.97e7 |
| Mayor celda intermedia, agregación 77x11 (millones de USD) | 5.83e10 | 6.79e6 |
| Celdas intermedias de la agregación 77x11 que coinciden | 35.5% | referencia |
| Celdas de valor agregado de la agregación 77x11 que coinciden | 1 de 847 | referencia |
| Celdas intermedias nativas positivas de 45 sectores puestas a cero en la exportación (todas las inferiores a 0.001) | 2.10 millones de 7.82 millones | ninguna |
| Celdas intermedias nativas positivas multiplicadas por 1e3 o 1e4 en la exportación | 0.70 millones | ninguna |

| Superficie | Condición afectada | Qué hacer |
|---|---|---|
| `load_icio_data()`, la calibración 77x11 incluida, cuadernos 61-65, `trade_reference_solutions.npz`, `trade_results_workbook.npz` | Cualquier magnitud, participación o elasticidad calculada sobre la tabla incluida | Trátelas como fixtures de regresión de software: las pruebas de paridad comparan puremacro con soluciones MATLAB de la **misma** tabla y siguen siendo internamente consistentes. No reporte sus salidas como estimaciones basadas en la OCDE. |
| `load_raw_45sector_icio()` sin `path` (4.3.0 y anteriores) | La lista de búsqueda por defecto empezaba por `computation/7_TIO_77c_vf/data_2020_SML.csv`, la exportación dañada, y la leía sin suma de verificación | La versión corregida busca primero los archivos limpios `ICIOextended/2020_SML.csv` y `2019_SML.csv`, comprueba el MD5 del archivo resuelto, rechaza la exportación dañada con `MRIOIntegrityError` (un `ValueError`) indicando las sumas de verificación de la publicación limpia y avisa ante sumas desconocidas. Pase explícitamente `2019_SML.csv` (MD5 `28cba31491177955445051d459053744`) o `2020_SML.csv`. |
| Lectores nativos (`load_oecd_icio_granular`, `puremacro.trade.mrio.read_oecd_native`) | Ninguna: leen el CSV regular OCDE 2023 por etiquetas | Úselos para trabajo empírico nuevo; registran la suma de verificación de la fuente en los metadatos. |

La tabla incluida se conserva sin cambios para que el contrato histórico de
paridad con MATLAB siga siendo externo. Sustituirla por una agregación limpia
cambiaría en silencio todos los fixtures y es una decisión del autor, no de una
versión. Véase [tablas MRIO nativas](trade_mrio.md) para las mediciones.

---

## 2026-09-20 — modelos estructurales, APIs afectadas en 4.2.0 y anteriores

**Corregidas o restringidas explícitamente en 4.3.0.** La revisión reprodujo
resultados numéricos y señales de éxito incorrectos en VFI, DSGE/Dynare y comercio
sobre el código de 4.2.0; la primera versión afectada depende de cada API.

- **DSGE de tercer orden:** recalcular reglas y simulaciones, momentos o
  verosimilitudes que dependen de pendientes de riesgo en modelos estocásticos
  no lineales. Las reglas de primero y segundo orden no contienen esas pendientes.
- **Paridad con Dynare:** repetir comparaciones de segundo orden con todos los
  tensores y etiquetas. La puntuación anterior podía omitir términos o aceptar
  referencias incompletas; no acreditaba paridad completa.
- **Proyecciones VFI:** exigir el residuo final solicitado, no solo la terminación
  del optimizador. Regenerar valores auxiliares con productividad distinta de uno
  o utilidad distinta de logaritmo. La exactitud fuera de la malla sigue pendiente.
- **Deep Macro:** repetir el entrenamiento y comprobar residuos fuera de la
  muestra; se corrigió la caché de activaciones usada en el gradiente.
- **DSGE con cambios de régimen:** no interpretar impactos, estabilidad o momentos
  de soluciones fallidas; ahora se reportan como no disponibles.
- **Comercio:** repetir contrafactuales CES y experimentos con recaudación o
  valoración inconsistente. Seleccionar `accounting="consistent"` para el modelo
  derivado de precios al productor y comprador; el modo histórico sigue por
  compatibilidad.

La política arancelaria selecciona bienestar de consumo basado en la función de
gasto mediante `metric="hicksian_ev"`. Los alias históricos conservan sus proxies.
La referencia es fija y los equilibrios se auditan; no se certifica un teorema
global de Nash. Las interfaces preliminares TOT/Alloc/TariffRec y de certificación
de teoremas siguen no disponibles por carecer de fundamento independiente.

Véase [el alcance y las referencias](STRUCTURAL_VALIDATION_STATUS.md),
[la contabilidad](trade_accounting.md), [el bienestar](trade_welfare.md) y
[la política arancelaria](trade_policy.md). Estas referencias delimitan qué
resultados se han validado y qué extensiones siguen pendientes.

---

## 2026-09-16 — tres estimadores, versiones hasta 4.0.1 inclusive

**Corregido en 4.0.2.** Se encontraron cuando un sitio de curso volvió a derivar
los números de puremacro por su cuenta: un filtro HP aplicado en el dominio del
tiempo, un MC2E en forma matricial y una cadena de Markov resuelta con 80 dígitos.

| Estimador | Qué estaba mal | No afectado cuando | Dirección del error |
|---|---|---|---|
| `theoretical_moments(hp_filter=...)` vía `dsge._moments.spectral_moments` (2.9.0–4.0.1) | Ponderaba la densidad espectral con la función de transferencia `H(w)` del filtro HP en lugar de su cuadrado | Momentos teóricos sin filtrar o con filtro pasa-banda, y momentos simulados de `stoch_simul(hp_filter=...)`, que filtran en el dominio del tiempo | Varianzas y autocovarianzas filtradas **sobreestimadas**: +22% de varianza (+10.5% de desviación estándar) para un AR(1) con rho = 0.9 y lambda = 1600; las volatilidades relativas y las correlaciones se movieron con ellas |
| `lp.lp_iv` (0.92.0–4.0.1), `lp.lp_state_dep_iv` (2.0.0–4.0.1), `lp.la_lp_iv` (4.0.0–4.0.1) | La varianza de la segunda etapa usaba el residuo `y - X_hat beta` (regresor ajustado) en lugar de `y - X beta` | Nunca para `se`, `lo`, `hi`; las estimaciones puntuales, la F de primera etapa, la F de Montiel Olea–Pflueger y los conjuntos de Anderson–Rubin eran correctos | **En cualquier sentido**, en `2 beta cov(u, v) + beta^2 var(v)`: las bandas nominales de 90% cubrieron el valor verdadero 100% de las veces con `beta = 1`, `cov(u, v) = 0.8`, y 69% con `beta = -1` |
| `vfi.markov_stationary` (0.92.0–4.0.1), `dsge.markov_switching.markov_stationary` (3.1.0–4.0.1) | Tomaba el eigenvector de `P'` del eigenvalor más cercano a 1 | Los estados se comunican con probabilidades muy por encima de ~1e-12, como en las mallas usuales de Tauchen y Rouwenhorst | Una mezcla arbitraria cuando varios eigenvalores redondean a 1: **error de 6e-5** en una cadena de dos estados con probabilidad de cambio de 1e-13, una **probabilidad de -0.23** en `tauchen(7, 0.995, 0.1, m=5)`; la copia de `dsge` regresaba una distribución uniforme sin avisar |

### Qué hay que volver a correr

- **Cualquier cuadro de momentos teóricos con filtro HP**, por ejemplo una
  comparación à la Kydland–Prescott de un modelo DSGE con los datos. Los momentos
  simulados eran correctos, así que un cuadro que mezclaba ambos comparaba cosas
  distintas.
- **Cualquier banda o estadístico t de `lp_iv`, `la_lp_iv` o `lp_state_dep_iv`.**
  Que fuera demasiado ancha o demasiado angosta depende del signo de
  `beta cov(u, v)`, así que no se corrige reescalando: hay que volver a estimar.
- **Distribuciones estacionarias de discretizaciones muy persistentes** (Tauchen
  con rho de alrededor de 0.95 o más y malla amplia) o de modelos con cambio de
  régimen markoviano con regímenes casi absorbentes.

---

## 2026-09-03 — doce más, versiones hasta 1.9.0 inclusive

**Corregido en 1.10.0.** Una auditoría posterior
preguntó, de cada fixture de estimador del paquete, *¿qué condición elimina este
fixture?* — la pregunta sobre la que giraban los siete defectos de arriba.
Encontró doce más, cada uno reproducido contra una verdad conocida antes de
tocar nada.

| Estimador | Qué estaba mal | No afectado cuando | Dirección del error |
|---|---|---|---|
| `cointegration_modern.dols` | Omitía el término contemporáneo `dX_t`, que es el que elimina el sesgo. El estimador era MCO con regresores extra. | `dX_t` no está correlacionado con `u_t` | El sesgo seguía al de MCO en todos los tamaños: **+0.036 frente a +0.030** en T=100, donde el DOLS correcto da −0.0005 |
| `cointegration_modern.dols` | Conservaba una fila cuyo rezago de `dX` faltante se **rellenaba con ceros**, fabricando un valor de regresor | `lags = 0` | Una observación de más, construida con un valor nunca medido |
| `cointegration_modern.fm_ols` | Construía la corrección de Phillips–Hansen sólo con `Lambda`, omitiendo el bloque `Sigma` de rezago 0 | `Omega` es proporcional a `Sigma` | Sobre-corregía. **ECM peor que MCO** en T=200, 800 y 3200 |
| `var.irf.gfevd` → `connectedness.spillover_index` | Dividía por un piso absoluto de `1e-12`, rompiendo una invarianza de escala que el estimando tiene por construcción | Ninguna varianza residual cae cerca del piso | La conectividad total pasó de **13.43 a 39.48** en un mismo VAR al reescalar una variable |
| `dsge.gensys` | `Impact` omitía el término `Pi N`: resolvía como si los errores de expectativas no respondieran a los choques | `Pi N = 0` | En `y_t = a E_t y_{t+1} + eps_t` (verdad `y_t = eps_t`) devolvía **`[0, -2]`**: la variable no respondía a su propio choque |
| `dsge.gensys` | Lanzaba excepción con cualquier modelo **sin raíces inestables**; la prueba de unicidad era vacua | El modelo tiene al menos una raíz inestable | `x_t = rho x_{t-1} + eps` no podía resolverse |
| `dsge.fertility_adj_costs` `irf`/`fevd` | Avanzaba el estado antes de aplicar `F`, pero los controles leen el estado *rezagado* | Sólo el horizonte 0 | Toda IRF de control **un periodo adelantada**: el `y(h)` reportado era el `y(h+1)` correcto |
| `did.sun_abraham` | `lo`/`hi` promediaban los extremos por cohorte en vez de usar el `se` agregado | Una sola cohorte por tiempo de evento | Bandas **`sqrt(K)` veces más anchas** — 1.73 con K=3, 1.37 con K=2 — e inconsistentes con el `se` de su propia fila |
| `inference.weak_iv.kleibergen_paap_f` | El sándwich HC0 dividía entre `n` dos veces mientras el pan invertía el producto cruzado crudo `Z'Z`, que ya lleva ambos factores | Nunca — toda llamada estaba afectada | Devolvía **exactamente `n^2` veces** el estadístico correcto: 235,470 frente a una verdad de 5.89 con n=200. Los umbrales de Stock-Yogo rondan 10, así que reportaba "instrumentos fortísimos" para todo conjunto de datos y **jamás podía diagnosticar un instrumento débil** |
| `inference.weak_iv.kleibergen_paap_f` | La función de influencia usaba `kron(Z_t, V_t)` donde el `vec` por columnas y el pan exigen `kron(V_t, Z_t)` | `k = 1` (un regresor endógeno) — ahí coinciden | Varianza robusta incorrecta con dos o más regresores endógenos |
| `inference.weak_iv.anderson_rubin_band` | Comparaba un estadístico F contra `chi2.ppf(ci, df)`, pero es `df*F` lo que es `chi2(df)` | `df = 1` — el valor por defecto y el único ejercitado | Corte `df` veces mayor. La banda nominal del 90% tenía **100.0% de cobertura** con df=2 y df=4 |
| `lp.panel_lp_dk` (vía `_focal_dk_se`) | El ancho de banda de Bartlett salía del número de filas `N*T`, pero el núcleo corre sobre la suma transversal de longitud `T` | `N = 1` | Ancho inflado por `N^(2/9)`: **7 -> 10 -> 13** con T=100 fijo al pasar N de 10 a 50 a 200. Agregar países cambiaba la autocorrelación supuesta |
| `lp.mean_group_panel_lp` | No ordenaba su entrada, mientras `lp_hac` construye rezagos con `.shift()` posicional | Quien llama ya pasaba un marco ordenado | En un panel cuya respuesta verdadera en h=1 es 0.8: **0.765 ordenado, 0.055 con las mismas filas barajadas**. Silencioso, y atenuado hacia cero |
| 25 sitios de valor-p en 16 módulos | `1 - cdf` en vez de la función de supervivencia | El valor-p supera ~1e-16 | Devolvían **exactamente 0.0** en la cola. Normal bilateral en \|z\| = 9 es 2.3e-19; chi-cuadrada 200 con 5 gl es 2.8e-41 |

### Qué hay que volver a correr

- **Cualquier estimación de `dols` o `fm_ols`.** Ambas estaban sesgadas
  sistemáticamente sobre los datos para los que existen. `fm_ols` quedaba más
  lejos de la verdad que MCO.
- **Cualquier índice de conectividad de Diebold–Yilmaz** calculado con el
  `identification="gfevd"` por defecto sobre datos cuyas varianzas residuales
  sean pequeñas en las unidades empleadas.
- **Cualquier IRF de `gensys`.** La matriz de impacto era incorrecta para todo
  modelo con comportamiento prospectivo, que son todos los que gensys atiende.
- **Cualquier IRF o FEVD del DSGE de fecundidad**, incluidas las figuras de
  `docs/research/fertility_bk_diagnosis/`. Desplace las sendas de control un
  periodo hacia atrás, o vuelva a correrlas.
- **Cualquier banda de estudio de eventos de Sun–Abraham** en un tiempo de
  evento con más de una cohorte. Las estimaciones puntuales y el `se` se
  sostienen; los intervalos eran demasiado anchos. **Corrección
  (2026-09-30):** el `se` no se sostenía; trataba como independientes
  cohortes que comparten unidades de control (véase la entrada del
  2026-09-30 sobre los errores estándar de Sun-Abraham). Las estimaciones
  puntuales sí.
- **Cualquier valor-p reportado como exactamente 0.0.** Nunca fue cero.

### Una nota sobre las pruebas

Cuatro artefactos distintos afirmaban el defecto de `gensys` en vez de
detectarlo: tres pruebas en `tests/test_dsge_gensys_coverage.py` y el caso de
validación `dsge.gensys_shock_impact_identity`, cuya propia cita enunciaba el
modelo **sin** su término `Pi eta_t` — álgebra correcta a partir de una premisa
truncada. La banda de `sun_abraham` contradecía el error estándar impreso a su
lado en la misma fila, así que ésa no requería referencia externa alguna. Y el
docstring de `FertilitySolution` describía `F` actuando sobre el estado actual
mientras el solver lo construye contra el rezagado; el bucle de IRF se escribió
siguiendo el docstring.

---

## 2026-09-02 — siete estimadores, versiones 0.92.0 a 1.8.0

**Corregido en 1.9.0.** Siete estimadores públicos devolvieron números
equivocados en todas las versiones desde 0.92.0 hasta 1.8.0 inclusive.
Los siete fallos comparten una forma: la respuesta equivocada era
*internamente consistente*, de modo que ninguna invariante que el paquete
verificaba podía detectarla, y en todos los casos el fixture de prueba
cumplía exactamente la condición bajo la cual el error desaparece.

### ¿Le afecta?

```python
import puremacro
puremacro.__version__          # < "1.9.0" → lea el cuadro
```

Si **publicó** un número de alguno de los siete estimadores de abajo con
una versión anterior a 1.9.0, vuelva a correrlo. Si sólo lo corrió sobre
datos que cumplen la columna "no afectado cuando", no hay nada que hacer.

| Estimador | Qué estaba mal | No afectado cuando | Dirección del error |
|---|---|---|---|
| `var.identify.proxy_svar` | El vector de impacto se devolvía en la métrica de `Sigma` y no en la de `Sigma^-1` — proporcional a `Sigma b_1`, no a `b_1`. El choque identificado era una mezcla de todos los choques estructurales. | `Sigma` es proporcional a la identidad (residuos i.i.d.) — ahí el error es exactamente cero | Crece con la estructura fuera de la diagonal de `Sigma`. 31% en un elemento de un DGP de 3 variables, con el patrón de signos relativos equivocado |
| `var.panel` | Importa el mismo `_proxy_impact_factory` | ídem | ídem |
| `inference.swamy_test` | La forma cuadrática se centraba en la media aritmética y no en `beta_bar_W`, la ponderada por precisión | Todas las unidades se estiman con la **misma** precisión | Sobre-rechaza la homogeneidad de pendientes, nunca sub-rechaza. Tamaño al 5% nominal: 0.050 con precisión igual, 0.078 con dispersión 2×, **0.975** con 0.1 frente a 3.0 |
| `garch.dcc_fit` | Los rendimientos crudos se estandarizaban con una volatilidad ajustada sobre los centrados, de modo que `Qbar` estimaba `mu_i · mu_j` | `mean="zero"` — **el valor por defecto**, e idéntico bit a bit al anterior sobre datos ya centrados. Sólo `mean="constant"` está afectado | Correlaciones atraídas hacia `m²/(m²+1)`, con `m = mu/sd`. Un valor verdadero de 0 se reportaba como **+0.94** con media 5 y desviación 1 |
| `state_space.simulation_smoother` | Los interceptos del modelo quedaban en el segundo paso de Durbin–Koopman, sumando `b` una segunda vez | `c` y `d` son ambos cero — todos los fixtures de la suite, e idéntico bit a bit ahí | Cada draw desplazado por el intercepto completo. Con `d = 5` los draws quedaban exactamente −5.0 de la media posterior, contra un error Monte Carlo de 0.012 |
| `var.wild_bootstrap_var` | Los draws fallidos se escribían en la pila de percentiles como la estimación puntual, sin contador ni advertencia | Ningún draw falló | Bandas demasiado **estrechas**, monótonamente en la fracción de fallos `f`; ancho cero una vez que `f ≥ 1−2a`. Peor precisamente cuando el `impact_fn` de `proxy_svar` falla por instrumento débil — la banda se estrechaba cuando debía ensancharse |
| `var.identify.rigobon_svar` | El bootstrap emparejaba bloques de residuos remuestreados con las etiquetas de régimen en orden de calendario, destruyendo la identificación en cada draw | Sólo la estimación puntual; lo que estaba mal es la **banda** | Bandas unas **8 veces más anchas** en un DGP con razón de varianzas verdadera 3.0 (los draws promediaban 1.14 y nunca superaron 1.50 en 500) |
| `var.estimate_var` | Aceptaba datos no finitos y devolvía coeficientes todo-NaN sin lanzar excepción | La entrada es finita | No es un número equivocado sino uno silencioso: `Sigma`, los residuos y toda IRF, FEVD, descomposición histórica y banda construidas sobre el ajuste salían todo-NaN y perfectamente bien formadas. Ahora es un `LinAlgError` con nombre |

Las derivaciones completas, las magnitudes medidas y por qué cada fixture
no podía alcanzar su propio error están en
[`CHANGELOG.md`](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md),
bajo 1.9.0, "Fixed — affects results published in every release from
0.92.0 to 1.8.0".

### Qué hay que volver a correr

- **Cualquier respuesta a impulso proxy-SVAR publicada** de `proxy_svar`
  o `var.panel`. Cambian la estimación puntual y la banda.
- **Cualquier banda de Rigobon.** La estimación puntual se sostiene; la
  banda no.
- **Cualquier rechazo de `swamy_test` en un panel con precisión desigual
  entre unidades** — muestras cortas mezcladas con largas, países
  pequeños con grandes. Ése es el caso normal, no el exótico.
- **Cualquier correlación de `dcc_fit(mean="constant")`.** La ruta por
  defecto `mean="zero"` no requiere nada.
- **Cualquier draw de `simulation_smoother` de un modelo con deriva de
  estado o intercepto de medición distintos de cero.**
- **Cualquier banda de `wild_bootstrap_var` cuya corrida haya reportado
  fallos de bootstrap** — cosa que, antes de 1.9.0, no reportaba. Si el
  estimador era `proxy_svar` con instrumento débil, suponga que la banda
  quedó demasiado estrecha.

### Hasta dónde llegó la corrección en 1.9.0

Se dice aquí en vez de dejar que un usuario lo descubra:

- **`matlab/` — eliminada en 4.0.0.** La caja de herramientas de MATLAB era
  una implementación separada, así que ninguna corrección de Python llegaba a
  ella. Antes de eliminarla se portaron dos estimadores a mano:
  `+puremacro/+var/proxy.m` arrastraba el mismo error de métrica en el SVAR
  con proxy y quedó corregido, y `+puremacro/+var/estimate.m` ahora falla ante
  entradas no finitas. Los otros cinco estimadores del cuadro anterior **nunca
  se auditaron allí**. Como una implementación paralela sin auditar de un
  estimador ya corregido es un riesgo permanente, y como esa caja de
  herramientas exigía software propietario que puremacro no requiere, se
  eliminó en 4.0.0 en lugar de mantenerla. **Si alguna vez ejecutó esa caja de
  herramientas, considere sus resultados no verificados y vuelva a calcularlos
  con el paquete de Python. Toda función de impulso-respuesta de SVAR con
  proxy que haya producido antes del 2026-09-02 es incorrecta.** El código
  permanece en el historial de git en la etiqueta `v3.4.0`.
- **Los cuadernos de este repositorio ya se volvieron a ejecutar.**
  `notebooks/14_tax_multiplier_three_ways`,
  `notebooks/17_identification_spec_curve`, sus gemelos `_es`,
  `notebooks/course/06_lp_narrativa_es` y la compilación de
  `playground/` muestran números posteriores a la corrección desde
  1.9.0. `notebooks/08_garch_volatility` no requirió cambio: llama a
  `dcc_fit(panel)`, y la ruta por defecto `mean="zero"` es idéntica bit
  a bit.
- **Los suyos no.** Un `.ipynb` versionado guarda las salidas de la
  corrida que lo produjo. Toda celda suya que muestre un resultado de un
  estimador de arriba, ejecutada antes de 1.9.0, sigue mostrando el
  número previo a la corrección hasta que la vuelva a ejecutar.

---

## Cómo se decide un aviso

Se emite un aviso cuando se cumplen **todas** estas condiciones:

1. Una versión publicada devolvió un resultado numéricamente equivocado
   de un estimador público, o un resultado sin la cobertura que declaraba.
2. El fallo fue silencioso — sin excepción, sin advertencia, sin una
   salida visiblemente mal formada.
3. Un usuario pudo plausiblemente haber publicado ese número.

Un error que lanza excepción, uno en una ruta no publicada y uno en un
auxiliar privado sin consecuencia pública son entradas del CHANGELOG, no
avisos.

La regla que se sigue es la de
[`CONTRIBUTING.md`](https://github.com/jalonso1979/puremacro/blob/main/CONTRIBUTING.md):
el paquete no sustituye un valor faltante por uno plausible, y no se
queda callado sobre un número que calculó mal.
