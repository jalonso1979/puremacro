> 🇬🇧 [English](../did.md) · 🇪🇸 Español

# Diferencias en Diferencias Modernas (DiD)

Las regresiones clásicas con efectos fijos bidireccionales (Two-Way Fixed Effects, TWFE) de la forma:

$$y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$

fallan sistemáticamente cuando el momento de adopción del tratamiento es **escalonado** (diferentes unidades reciben el tratamiento en distintos períodos) y los efectos del tratamiento son **heterogéneos** entre cohortes o dinámicos en el tiempo (Goodman-Bacon 2021, de Chaisemartin y D'Haultfœuille 2020). Bajo efectos heterogéneos, TWFE utiliza implícitamente a unidades ya tratadas como controles para unidades tratadas posteriormente, generando ponderaciones negativas que pueden llegar a invertir el signo de la estimación.

`puremacro.did` implementa la suite completa de estimadores robustos a la heterogeneidad en Python puro (solo numpy / scipy / pandas), ofreciendo inferencia por bootstrap de panel, agregaciones dinámicas para estudios de eventos y exportación directa de tablas para publicaciones.

Cuando no existe un antes/después limpio y la identificación descansa en condicionar sobre controles de alta dimensión, véase [`docs/es/causal_dml.md`](causal_dml.md) para el modelo parcialmente lineal con aprendizaje automático doble/desesgado.

---

## Panorama de estimadores

| Estimador | Función | Referencia clave | Estrategia econométrica |
|---|---|---|---|
| **Callaway y Sant'Anna** | `callaway_santanna` | Callaway y Sant'Anna (2021, *J. Econometrics*) | Efectos de grupo y tiempo $ATT(g, t)$ con grupos de control limpios (nunca tratados o no tratados aún) |
| **Sun y Abraham** | `sun_abraham` | Sun y Abraham (2021, *J. Econometrics*) | Estudio de eventos con ponderación por cohortes para aislar cambios composicionales |
| **Borusyak, Jaravel y Spiess** | `borusyak_jaravel_spiess` | Borusyak, Jaravel y Spiess (2024, *Rev. Econ. Stud.*) | Estimador de imputación: ajusta efectos fijos sobre observaciones no tratadas y proyecta contrafactuales |
| **de Chaisemartin y D'Haultfœuille** | `cdh_did` | de Chaisemartin y D'Haultfœuille (2020, *AER*) | Estimador de cambiantes $DID_M$ / $DID_M^\ell$ con prueba placebo |
| **DiD Sintético (SDID)** | `synthetic_did` | Arkhangelsky, Athey et al. (2021, *AER*) | Doble ponderación: pesos de unidad $\omega$ para ajustar pre-tendencias + pesos temporales $\lambda$ |
| **SDID multi-cohorte** | `sdid_multi_cohort` | Arkhangelsky et al. (2021); Roth et al. (2023, *J. Econometrics*) | `synthetic_did` por cohorte sobre una ventana de donantes no tratados, ponderado por tamaño de cohorte |
| **DiD robusto a desbordamientos** | `spatial_did` | Clarke (2017); Berg, Reisinger y Streitz (2021); Butts (2023) | Un coeficiente por anillo de distancia alrededor de las unidades tratadas, con las unidades más allá del anillo exterior como único control |

Conviven dos convenciones de entrada. `callaway_santanna`, `sun_abraham`, `borusyak_jaravel_spiess`, `synthetic_did` y `spatial_did` reciben un **DataFrame en formato largo** con los nombres de columna en `unit=`, `time=`, `outcome=`, `treat_time=` (`treat_time` es el primer período de tratamiento de cada unidad, `NaN` para las nunca tratadas). `spatial_did` necesita además la geografía, por encima de esos cuatro nombres de columna: `coords=` (un DataFrame indexado por identificador de unidad cuyas dos primeras columnas son `[lat, lon]`, o un mapeo `id -> (lat, lon)`) o una asignación de anillos construida de antemano y pasada como `assignment=`. `cdh_did` y `sdid_multi_cohort` reciben **cuatro arrays 1-D alineados** `(y, treatment, panel_id, time_id)`, donde `treatment` es el estado de tratamiento 0/1 de cada fila.

---

## 1. Callaway y Sant'Anna (2021)

Estima los efectos medios del tratamiento para cada cohorte $g$ (año de adopción) en cada período $t$, denotados como $ATT(g, t)$, frente al período base universal $g - 1$, y los agrega en un perfil de estudio de eventos relativo al momento de intervención $e = t - g$. El bloque siguiente construye un panel escalonado sintético que reutilizan todos los bloques posteriores:

```python
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

# Panel escalonado sintético: 60 condados x 12 años. Las cohortes adoptan en
# 2004 y 2007; un tercer grupo nunca adopta. Efecto real = 1.0 + 0.2 * (años
# desde la adopción); los resultados llevan efectos fijos de condado y año.
rng = np.random.default_rng(0)
cohorts = np.array([2004.0, 2007.0, np.nan])
rows = []
for i in range(60):
    g = cohorts[i % 3]
    alpha_i = rng.normal()
    for year in range(2000, 2012):
        e = year - g if not np.isnan(g) else -1.0
        tau = 1.0 + 0.2 * e if e >= 0 else 0.0
        rows.append({
            "county_id": i, "year": year, "first_treated_year": g,
            "employment": alpha_i + 0.1 * (year - 2000) + tau + rng.normal(scale=0.3),
        })
df = pd.DataFrame(rows)

res_cs = callaway_santanna(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control="never_treated",   # o "not_yet_treated" (se acepta control_group= como alias)
    n_boot=500,
    ci=0.95,
)

print(res_cs.summary())
print(res_cs.att_event_study.head())
print(res_cs.to_latex())
print(res_cs.to_typst())
```

### Agregación: pesos por tamaño de cohorte, como en el artículo

Todos los agregados ponderan las cohortes por su **tamaño**. Si $n_g$ es el número de unidades de la cohorte $g$, la proporción $n_g / \sum_{g'} n_{g'}$ estima $P(G = g \mid G \le T)$. La numeración de ecuaciones es la de la versión arXiv del artículo (1803.09015v4, sección 3, pp. 15–19).

- **Estudio de eventos** (ec. 3.4): $\theta_{es}(e) = \sum_g \mathbf{1}\{g + e \le T\}\, P(G = g \mid G + e \le T)\, ATT(g, g + e)$, el promedio ponderado por tamaño de las cohortes observadas $e$ períodos después de adoptar. Los pre-períodos usan los mismos pesos.
- **ATT global**, que se elige con `aggregation=`:

| `aggregation=` | Estimando | Definición |
|---|---|---|
| `"group"` (por defecto) | $\theta^O_{sel}$, ec. 3.11 | $\sum_g P(G = g \mid G \le T)\,\theta_{sel}(g)$, donde $\theta_{sel}(g)$ (ec. 3.7) es la media de las celdas post-tratamiento de la cohorte $g$. El artículo lo recomienda como resumen general: el efecto medio que experimentan las unidades alguna vez tratadas, el análogo del ATT 2×2. |
| `"simple"` | $\theta^O_W$, ec. 3.10 | Cada celda post-tratamiento ponderada por el tamaño de su cohorte. Da más peso a las cohortes tempranas, que tienen más celdas post-tratamiento. |
| `"dynamic"` | $\theta^O_{es}$, ec. 3.12 | Media de $\theta_{es}(e)$ sobre $e \ge 0$. |
| `"calendar"` | $\theta^O_c$, ec. 3.12 | Media, sobre los períodos con tratamiento, de $\theta_c(t)$ (ec. 3.8), el efecto ponderado por tamaño de las cohortes tratadas en $t$. |
| `"unweighted"` | ninguno | La regla de puremacro 4.3.0 y anteriores: cada cohorte cuenta una vez en el estudio de eventos y `att_overall` es la media simple de las celdas post-tratamiento. **No** es un estimando de Callaway–Sant'Anna; úsela solo para reproducir números antiguos. |

Los cinco resúmenes coinciden cuando todos los $ATT(g, t)$ son iguales. Se separan cuando los efectos difieren entre cohortes y las cohortes difieren en tamaño. En un panel sin ruido con una cohorte de 10 unidades (efectos 1, 1.5, 2, 2.5) y otra de 40 (efectos 3, 4), $\theta_{es}(0) = 2.6$ y $\theta^O_{sel} = 3.15$, mientras que la regla de cohortes iguales da 2.0 y 2.33 (caso de validación `did.callaway_santanna_aggregations_match_cs2021_equations`).

**Inferencia.** Los errores estándar salen de un bootstrap de panel que remuestrea unidades completas. Cada réplica reestima todos los $ATT(g, t)$ *y* los tamaños de cohorte, así que el `se` de cada agregado incluye la covarianza entre cohortes que comparten controles y la incertidumbre de estimar los pesos (el Corolario 2 del artículo). `lo`/`hi` son bandas percentiles puntuales.

Atributos principales de `CallawaySantannaResult`:
- `att_gt`: DataFrame con las estimaciones $ATT(g, t)$ y sus errores estándar bootstrap (columnas `g, t, event_time, att, se, lo, hi`).
- `att_event_study`: el estudio de eventos $\theta_{es}(e)$ de la ec. 3.4; `n_cohorts` es el número de cohortes que identifican cada horizonte.
- `att_overall`, `att_overall_se`, `att_overall_lo`, `att_overall_hi`: el resumen global elegido con `aggregation` ($\theta^O_{sel}$ por defecto), con su error estándar bootstrap y su banda percentil.
- `att_group`: $\theta_{sel}(g)$ por cohorte (columnas `g, n_g, weight, att, se, lo, hi`); `weight` es el peso de la cohorte en $\theta^O_{sel}$.
- `overall_aggregations`: los cinco resúmenes globales lado a lado (columnas `aggregation, estimand, att, se, lo, hi`).
- `event_study_vcov`: la matriz de covarianzas bootstrap de los coeficientes del estudio de eventos. Páselo como `sigma=` a `honest_did`, que de lo contrario usa solo la diagonal.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Métodos de exportación del estudio de eventos (sin columna de índice); `.plot()` lo dibuja con su banda de confianza.

---

## 2. Sun y Abraham (2021)

Sun y Abraham muestran que los coeficientes de una regresión dinámica de efectos fijos bidireccionales pueden contaminarse con efectos de otros períodos relativos. Su estimador ponderado por interacción (IW) estima en cambio cada efecto por cohorte $CATT_{e,\ell}$ y los promedia en cada período relativo $\ell$ con la participación muestral de cada cohorte entre las observadas en ese período (su ec. 27). Sin covariables y con controles nunca tratados, Sun y Abraham señalan que su estimador coincide con el de Callaway y Sant'Anna (arXiv:1804.05785, p. 24): el estudio de eventos de `sun_abraham` es el $\theta_{es}(e)$ de la ec. 3.4 anterior. Los dos estimadores **no** difieren en cómo ponderan las cohortes; ambos usan las participaciones de cohorte.

`sun_abraham` reutiliza las estimaciones grupo-tiempo *y las réplicas bootstrap conjuntas* de `callaway_santanna`. Su `se` del estudio de eventos es la desviación estándar bootstrap del propio agregado, con las participaciones de cohorte reestimadas en cada réplica. Eso recoge la covarianza entre cohortes que comparten los mismos controles y el término de estimación de pesos de la Proposición 6 de Sun y Abraham. `lo`/`hi` son `att ∓ z·se`. Con la misma `seed`, su columna `se` es igual a la de `callaway_santanna`.

Hasta la versión 4.3.0 el `se` agregado era $\sqrt{\sum_g w_g^2 se_g^2}$, que trata las cohortes como independientes. Según el proceso de error, quedaba entre 0.6 y 1.4 veces la verdad de Monte Carlo. `att_overall` usa por defecto `aggregation="simple"`, que pondera cada celda post-tratamiento por el tamaño de su cohorte ($\theta^O_W$, ec. 3.10, el valor que `sun_abraham` siempre ha devuelto). `aggregation="dynamic"` da el $\nu_g$ propio de Sun y Abraham para $g$ = los períodos post-tratamiento. `att_overall_se`, `att_overall_lo` y `att_overall_hi` dan su error estándar bootstrap conjunto y su banda:

```python
from puremacro.did import sun_abraham

res_sa = sun_abraham(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    ci=0.90,
)
print(res_sa.summary())
print(res_sa.to_markdown())
```

---

## 3. Estimador de imputación de Borusyak, Jaravel y Spiess (2024)

El estimador BJS es asintóticamente eficiente bajo el supuesto de tendencias paralelas y opera en tres pasos intuitivos:
1. **Ajuste**: Estima los efectos fijos de unidad y tiempo utilizando *únicamente* las observaciones no tratadas ($D_{it} = 0$: unidades nunca tratadas y filas pre-tratamiento de las unidades que llegan a tratarse).
2. **Imputación**: Proyecta los resultados contrafactuales $\hat{y}_{it}(0)$ para las celdas tratadas.
3. **Promedio**: Calcula el efecto como $\hat{\tau}_{it} = y_{it} - \hat{y}_{it}(0)$ y agrega sobre el horizonte de eventos (solo $e \ge 0$: BJS evalúa $\hat\tau$ en celdas tratadas, por lo que no hay filas de pre-tendencias). `att_overall` pondera por igual cada celda tratada.

Una celda tratada solo está identificada si su período y su unidad tienen al menos una observación no tratada. En un panel **sin unidades nunca tratadas**, ningún período desde la adopción de la última cohorte tiene observaciones no tratadas, así que su efecto fijo temporal no puede estimarse: `borusyak_jaravel_spiess` lanza por defecto un `ValueError` que nombra esos períodos, y con `unidentified="drop"` avisa y excluye esas celdas de todos los agregados.

```python
from puremacro.did import borusyak_jaravel_spiess

res_bjs = borusyak_jaravel_spiess(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    n_boot=500,
)
print(res_bjs.summary())
print("ATT global:", res_bjs.att_overall)
```

---

## 4. $DID_M$ de de Chaisemartin y D'Haultfœuille (2020)

`cdh_did` compara las unidades que **cambian** a tratadas entre $t-1$ y $t$ con las unidades cuyo estado permanece en 0 durante la misma ventana, evitando los pesos negativos de TWFE. Reporta el $DID_M$ instantáneo, los $DID_M^\ell$ de largo plazo para los horizontes $\ell$ y el valor $p$ del placebo de cambiantes (tendencia previa al cambio de los cambiantes frente a las unidades estables). Este estimador usa la convención de cuatro arrays:

```python
from puremacro.did import cdh_did

# Estado de tratamiento 0/1 por fila (un first_treated_year NaN compara como False -> 0)
treated = (df["year"] >= df["first_treated_year"]).astype(int).to_numpy()

res_cdh = cdh_did(
    df["employment"].to_numpy(), treated,
    df["county_id"].to_numpy(), df["year"].to_numpy(),
    horizons=(1, 2, 3), n_boot=200, seed=0,
)
print(res_cdh.summary())
print(res_cdh.to_markdown())   # columnas [estimand, horizon, att, se]
```

---

## 5. Diferencias en diferencias sintéticas (SDID)

Arkhangelsky et al. (2021) unifican el control sintético y las diferencias en diferencias:
- A diferencia del control sintético tradicional, SDID es invariante a desplazamientos aditivos de nivel entre unidades: ambos problemas de pesos incluyen los interceptos $\omega_0$, $\lambda_0$ del artículo, de modo que sumar una constante a la trayectoria de cualquier unidad no altera $\hat\tau$. Un desplazamiento común de los períodos mueve $\hat\tau$ solo a través del nivel de ruido $\hat\sigma$ que fija la penalización ridge (véase abajo); una tendencia lineal común no altera $\hat\sigma$ y, por tanto, tampoco $\hat\tau$.
- A diferencia del DiD clásico, no exige tendencias paralelas entre el grupo tratado y todas las unidades de control; en su lugar, optimiza pesos no negativos $\omega_i \ge 0$ para alinear las tendencias previas y pesos temporales $\lambda_t \ge 0$ para ponderar los períodos pre-tratamiento más relevantes:

$$\hat{\tau}^{\text{SDID}} = \arg\min_{\tau, \mu, \alpha, \beta} \sum_{i=1}^N \sum_{t=1}^T \left( y_{it} - \mu - \alpha_i - \beta_t - \tau W_{it} \right)^2 \hat{\omega}_i \hat{\lambda}_t$$

`synthetic_did` maneja una **única cohorte de tratamiento** (un período de adopción común, `treat_time` igual para todas las unidades tratadas y `NaN` para los donantes) y requiere un **panel balanceado**: una celda `(unit, time)` ausente lanza un `ValueError` que la identifica.

**Pesos.** La numeración de ecuaciones y algoritmos es la de arXiv:1812.09970v4. Los pesos de unidad resuelven la ec. (2.1) con penalización ridge $\zeta_\omega^2 T_{pre}\lVert\omega\rVert^2$, $\zeta_\omega = (N_{tr} T_{post})^{1/4}\hat\sigma$; los pesos temporales resuelven la ec. (2.3) con la penalización mínima $\zeta_\lambda = 10^{-6}\hat\sigma$ de la nota 3. $\hat\sigma$ (ec. 2.2) es la desviación estándar de las primeras diferencias pre-tratamiento de los controles, $Y_{i,t+1} - Y_{it}$, en torno a su media global, como en el paquete de R `synthdid`; `noise_level=` permite fijarla. Cada problema de pesos se resuelve en unidades de la raíz del cuadrado medio de los datos y se comprueba contra sus condiciones de optimalidad (KKT), de modo que $\hat\tau(c\,y) = c\,\hat\tau(y)$ para cualquier cambio de escala del resultado. Si el optimizador no alcanza el óptimo, emite una advertencia y devuelve los mejores pesos factibles que encontró. Hasta la versión 4.3.0, una resolución de pesos que SLSQP marcaba como fallida devolvía pesos uniformes sin avisar, lo que convertía SDID en un DiD simple cuando el resultado se medía en unidades grandes; con los datos de California del artículo devolvía la estimación DiD, −27.35.

**Inferencia.** El estimador de varianza se elige con `se_method=` (sección 5 del artículo). Los intervalos son gaussianos, $\hat\tau \pm z_{1-\alpha/2}\,\text{se}$ (ec. 5.1). `n_boot` es el número de réplicas $B$; cada réplica reutiliza los $\zeta_\omega$, $\zeta_\lambda$ de la muestra completa, como hace `synthdid`.

| `se_method=` | Algoritmo | Cuándo |
|---|---|---|
| `"placebo"` | 4 | Por defecto con **una o dos unidades tratadas**. Reasigna el tratamiento a $N_{tr}$ controles extraídos sin reemplazo y reestima SDID solo con los controles. Supone que las unidades comparten una misma distribución del ruido (homocedasticidad); requiere más controles que unidades tratadas. |
| `"bootstrap"` | 2 | Por defecto con **tres o más unidades tratadas**. Remuestrea todas las unidades, tratadas y de control, y vuelve a extraer las muestras que carecen de alguno de los dos grupos. Con una sola unidad tratada no está bien definido (no puede remuestrear el ruido de la unidad tratada) y con dos subcubre; en ambos casos emite una advertencia. |
| `"jackknife"` | 3 | Deja fuera una unidad cada vez, con $\hat\omega$, $\hat\lambda$ fijos. Rápido y conservador; NaN con una sola unidad tratada. |

`"auto"` (el valor por defecto) elige el estimador placebo con una o dos unidades tratadas (si hay más controles que unidades tratadas) y el bootstrap en otro caso. Hasta la versión 4.3.0 la única opción era un bootstrap sobre los donantes que mantenía fijas las unidades tratadas, con un intervalo percentil. Con una sola unidad tratada, su intervalo nominal del 90% cubría el valor verdadero entre el 50% y el 67% de las veces.

Cobertura de Monte Carlo de los nuevos intervalos nominales del 90%:

| Diseño | $N_{tr}$ | Por defecto | Cobertura (e.e. MC) | se medio / desviación verdadera |
|---|---|---|---|---|
| Notebook 29 (15 donantes, modelo de factores, ruido con desviación 0.4), 400 réplicas | 1 | placebo | 0.892 (0.015) | 0.244 / 0.234 |
| Efectos fijos bidireccionales, iid N(0,1), 30 controles, $T = 20$, $T_{pre} = 15$, 400 réplicas | 1 | placebo | 0.905 (0.015) | 0.598 / 0.563 |
| Igual, 200 réplicas | 2 | placebo | 0.885 (0.023) | 0.425 / 0.422 |
| Igual, 200 réplicas | 3 | bootstrap | 0.885 (0.023) | 0.356 / 0.340 |
| Igual, 200 réplicas | 5 | bootstrap | 0.910 (0.020) | 0.304 / 0.279 |
| Ejemplo de abajo (8 de 40 tratadas), 200 réplicas | 8 | bootstrap | 0.910 (0.020) | 0.062 / 0.057 |

Con dos unidades tratadas el bootstrap cubrió solo 0.790 (0.029); por eso `"auto"` cambia a partir de tres.

**Proposición 99 de California.** La tabla 1 del artículo (p. 8) da SDID −15.6 (error estándar placebo 8.4), SC −19.6 y DID −27.3. Con el mismo panel (los datos `california_prop99` del paquete de R `synthdid`), `synthetic_did` devuelve $\hat\tau = -15.61$. Los pesos temporales recaen en 1986–1988 (0.366, 0.206, 0.427), y Nevada, New Hampshire y Connecticut reciben los mayores pesos de unidad. El error estándar placebo es una cantidad de Monte Carlo: con $B = 200$ vale 10.3 con `seed=0` y va de 7.96 a 10.91 con las semillas 0–19, en torno a su límite de 9.37 cuando $B \to \infty$; el 8.4 del artículo es una de esas extracciones. El óptimo exacto de SC con la configuración del artículo es −19.51. El −19.6 del artículo sale del optimizador Frank–Wolfe de `synthdid`, que se detiene antes del óptimo; SDID no se ve afectado (Frank–Wolfe −15.60).

```python
from puremacro.did import synthetic_did

# Una sola cohorte de reforma: 8 estados adoptan en el trimestre 12, 32 nunca lo hacen.
rng = np.random.default_rng(1)
rows = []
for s in range(40):
    reform = s < 8
    alpha_s = rng.normal(scale=2.0)
    for q in range(24):
        effect = 0.8 if (reform and q >= 12) else 0.0
        rows.append({
            "state": f"S{s:02d}", "quarter": q,
            "reform_quarter": 12.0 if reform else np.nan,
            "gdp_growth": alpha_s + 0.05 * q + effect + rng.normal(scale=0.3),
        })
panel_sdid = pd.DataFrame(rows)

res_sdid = synthetic_did(
    panel_sdid,
    unit="state",
    time="quarter",
    outcome="gdp_growth",
    treat_time="reform_quarter",   # período de adopción por unidad, NaN para donantes
    n_boot=200,
    seed=0,
)

print(res_sdid.summary())
print("Principales unidades donantes:\n", res_sdid.omega[res_sdid.omega > 0.05])
print(res_sdid.lambda_w.round(3))
fig = res_sdid.plot()   # media tratada vs trayectoria sintética ponderada por omega
```

Para adopción escalonada en varias cohortes, use `sdid_multi_cohort`. Ejecuta `synthetic_did` una vez por cohorte de adopción y promedia las estimaciones con pesos por tamaño de cohorte. El grupo de donantes de cada cohorte permanece no tratado durante toda su ventana SDID: `control="never_treated"` usa las unidades nunca tratadas sobre el panel completo, `control="not_yet_treated"` admite también unidades tratadas más tarde pero trunca la ventana en su primera fecha de adopción, y el valor por defecto `"auto"` elige donantes nunca tratados cuando hay al menos dos. Su `se` sale de un bootstrap sobre unidades, tratadas y de control, que repite todo el procedimiento en cada réplica. Como el algoritmo 2, necesita varias unidades tratadas y emite una advertencia cuando el diseño solo tiene una o dos (en ese caso use `synthetic_did` y su estimador placebo). Como `cdh_did`, usa la forma de cuatro arrays:

```python
from puremacro.did import sdid_multi_cohort

res_multi = sdid_multi_cohort(
    df["employment"].to_numpy(),        # resultado
    treated,                            # estado de tratamiento 0/1 por fila
    df["county_id"].to_numpy(),         # identificador de unidad
    df["year"].to_numpy(),              # identificador de tiempo
    aggregation="att_g_t",
    control="auto",
    n_boot=100,
    seed=0,
)
print(res_multi.summary())
print(res_multi.att_g_t)                # una fila por cohorte
print(res_multi.to_markdown())          # tabla por cohorte más el agregado
```

---

## 6. DiD robusto a desbordamientos: anillos de exposición

Una planta abre en un condado y contrata en los tres vecinos; un estado sube su salario mínimo y los compradores cruzan la frontera. En ambos casos el tratamiento alcanza a unidades que el diseño llama controles, y el supuesto SUTVA falla por partida doble: el grupo de comparación está parcialmente tratado, así que el efecto directo queda subestimado, y el desbordamiento — que suele ser el objeto interesante — resulta invisible para una regresión que no tiene ningún coeficiente para él.

`spatial_did` implementa la solución estándar (Clarke 2017; Berg, Reisinger y Streitz 2021; Butts 2023): particionar las unidades **no tratadas** en anillos de distancia alrededor de las tratadas, dar a cada anillo su propio coeficiente y conservar como grupo de comparación únicamente las unidades más allá del anillo exterior. Con adopción común y $\text{Post}_t = \mathbf{1}\{t \ge t_0\}$:

$$y_{it} = \mu_i + \tau_t + \sum_{r=0}^{R} \delta_r\, \text{Ring}^r_{it} + \varepsilon_{it}$$

$\text{Ring}^0_{it} = 1$ cuando la unidad $i$ está ella misma tratada en $t$; $\text{Ring}^r_{it} = 1$ cuando $i$ no está tratada en $t$ y su distancia a la unidad tratada más cercana cae en la banda $r$. La categoría omitida es "no tratada y más allá de `rings[-1]`", de modo que cada $\delta_r$ se lee contra esas unidades. El anillo 0 es **absorbente**: una unidad tratada junto a otra unidad tratada permanece en el anillo 0, así que $\delta_0$ es el efecto *total* sobre las tratadas — su efecto directo más el desbordamiento que recibe de otras unidades tratadas — y coincide con el ATT directo puro solo cuando no hay dos unidades tratadas a menos de `rings[-1]` una de otra. $\delta_1 \dots \delta_R$ son los desbordamientos.

El DiD ingenuo que agrupa a todas las unidades no tratadas en el control es exactamente la regresión corta, así que, con $M$ la proyección interna bidireccional, Frisch-Waugh (1933) da

$$\hat\delta_{\text{ingenuo}} = \delta_0 + \sum_{r=1}^{R} \theta_r \delta_r, \qquad \theta_r = \frac{(M\,\text{Ring}^0)'(M\,\text{Ring}^r)}{(M\,\text{Ring}^0)'(M\,\text{Ring}^0)}.$$

Los indicadores de anillo son mutuamente excluyentes y se encienden a la vez, así que $\theta_r < 0$ en cualquier diseño realista: un desbordamiento *positivo* sesga la estimación ingenua *hacia abajo*. El resultado reporta `naive_att`, `direct_effect` y `contamination = naive_att - direct_effect = Σ θ_r δ_r`, y la identidad se cumple con precisión de máquina — pero solo dentro de la única proyección interna que ejecuta `spatial_did`. No dice nada sobre una regresión TWFE ingenua ajustada en otra muestra, algo que importa en cuanto se descartan filas (`n_obs_dropped`, que imprime `summary()`).

### 6.1 Construir los anillos

La banda 1 es el intervalo cerrado $[0, \texttt{rings[1]}]$ y la banda $r \ge 2$ es el semiabierto $(\texttt{rings[r-1]}, \texttt{rings[r]}]$. Todo lo que quede más allá de `rings[-1]` — incluido el $+\infty$ de "todavía no se ha tratado nada cerca" — cae en el código de control $R+1$.

Bajo adopción escalonada el indicador de anillo varía en el tiempo, y el valor por defecto `ring_timing="already_treated"` mide la distancia a la unidad *ya* tratada más cercana en $t$: así todo indicador de anillo es idénticamente cero antes de que ocurra cualquier tratamiento cercano, y el período previo sigue siendo una línea base limpia. `ring_timing="ever_treated"` enciende en cambio el anillo en la fecha de adopción de la unidad tratada más cercana. Es la opción correcta cuando la anticipación es el mecanismo, y contamina mecánicamente el período previo en caso contrario; además ignora la exposición a una unidad tratada más cercana en el tiempo pero más lejana en el espacio. Los empates nunca cambian un código de anillo — la distancia es un mínimo —, solo afectan a la identidad reportada en `nearest_treated`, donde gana la cohorte más temprana y, dentro de una cohorte, la primera unidad en el orden de `ids`.

`exposure_rings` devuelve un `RingAssignment` que puede inspeccionar antes de estimar nada. El ejemplo usa los puntos internos de condado del Censo que incluye el paquete:

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_county_centroids
from puremacro.did import exposure_rings, spatial_did

# Geografía real: puntos internos del Censo para cuatro estados del Corn Belt.
counties = load_us_county_centroids()
region = counties[counties["state"].isin(["IA", "IL", "MO", "NE"])]
coords = region[["lat", "lon"]]          # latitud primero: el estándar de haversine

# 20 condados reciben una planta en 2015; los otros 389 nunca.
rng_sp = np.random.default_rng(7)
plants = rng_sp.choice(coords.index.to_numpy(), size=20, replace=False)
plant_year = pd.Series(np.where(coords.index.isin(plants), 2015.0, np.nan),
                       index=coords.index)
years = list(range(2011, 2021))

ra = exposure_rings(coords, treat_time=plant_year, times=years,
                    rings=(0.0, 40.0, 80.0, 150.0))
print(ra.summary())
print(ra.to_frame("counts"))       # ring, label, lo_edge, hi_edge, n_units, n_unit_periods
print(ra.unit_frame().head())      # una fila por unidad: distancia, anillo, anillo de entrada, fecha
```

Lea `counts` con cuidado. `n_unit_periods` particiona las $N \times T$ celdas y por tanto suma `len(ra.frame)`; `n_units` cuenta las unidades que *alguna vez* estuvieron en el anillo y **no** suma $N$ cuando los anillos varían en el tiempo — bajo `already_treated` toda unidad está en el anillo de control durante el período previo, y por eso la fila de control reporta las 409. `unit_frame()` colapsa a una fila por unidad y lanza un error cuando alguna unidad camina hacia dentro atravesando dos anillos (`is_static=False`, `n_switchers > 0`); use `.frame` en ese caso. `ra.plot()` dibuja el histograma de la distancia a la unidad tratada más cercana con los puntos de corte encima — corte donde la distribución es delgada, nunca a través de una moda.

### 6.2 Estimar

`cutoff_km` es **obligatorio** con el valor por defecto `cov_type="conley"` y deliberadamente no tiene valor por defecto: el radio de Conley es un supuesto sobre hasta dónde llega el campo de errores, y resolverlo en silencio enterraría esa decisión. `2 * rings[-1]` es el punto de partida natural, porque dos unidades no tratadas pueden estar cada una a menos de `rings[-1]` de la misma unidad tratada y aun así separadas por `2 * rings[-1]`.

```python
# Un resultado sintético sobre esa geografía real: efecto de 1.0 en el condado
# tratado y desbordamiento decreciente 0.5 / 0.25 / 0.10 en los tres anillos.
unit_ring = ra.unit_frame().set_index("unit")["ring"]
true_effect = unit_ring.map({0: 1.0, 1: 0.5, 2: 0.25, 3: 0.10, 4: 0.0})
mu = pd.Series(rng_sp.normal(size=len(coords)), index=coords.index)
tau = pd.Series(rng_sp.normal(scale=0.2, size=len(years)), index=years)
panel_sp = pd.DataFrame([
    {"county": c, "year": y, "plant_year": plant_year[c],
     "log_emp": mu[c] + tau[y] + true_effect[c] * (y >= 2015)
                + 0.25 * rng_sp.normal()}
    for c in coords.index for y in years
])

res_sp = spatial_did(
    panel_sp, unit="county", time="year", outcome="log_emp",
    treat_time="plant_year", coords=coords,
    rings=(0.0, 40.0, 80.0, 150.0),
    cutoff_km=300.0,        # = 2 * rings[-1]; no hay valor por defecto
    placebo_periods=2,
)
print(res_sp.summary())
```

La inferencia es por defecto el HAC espacio-temporal de Conley (1999): un núcleo de distancia entre unidades dentro de cada período y un núcleo de Bartlett entre períodos hasta `time_lags` (la regla de ancho de banda de Driscoll-Kraay cuando se deja en `None`). `cov_type="cluster"` está disponible y **sub-cubre aquí**: los anillos son una función determinista de la geografía, así que regresor y error están ambos correlacionados espacialmente, y agrupar por unidad descarta exactamente los términos fuera de la diagonal $K(d_{ij}) u_i u_j$ que el diseño induce. El error de Conley solo es válido bajo asintóticas de dominio creciente — la región se expande, el campo de mezcla decae, el radio crece despacio — y no dice nada bajo asintóticas de relleno. El tamaño muestral honesto es `n_treated_clusters`, el número de componentes conexas del grafo de unidades tratadas al radio `cutoff_km`; en la ejecución anterior 20 condados tratados en una región contigua forman apenas **dos** experimentos espacialmente separados con un radio de 300 km, `spatial_did` avisa y `summary()` lo imprime. Lea los valores p como indicativos.

### 6.3 Leer el resultado

```python
print(res_sp.ring_table[["ring", "label", "n_units", "n_obs",
                         "effect", "se", "p", "dropped"]].round(3))
print(f"ingenuo {res_sp.naive_att:+.3f} = directo {res_sp.direct_effect:+.3f} "
      f"+ contaminación {res_sp.contamination:+.3f};  theta {res_sp.theta.round(3)}")
print(bool(abs(res_sp.naive_att
               - (res_sp.direct_effect + res_sp.theta @ res_sp.coef[1:])) < 1e-12))

print(res_sp.att_by_ring_es.round(3))                          # estudio de eventos, un número por anillo
print(res_sp.event_study.head().round(3))                      # la trayectoria dinámica completa
print(res_sp.pretrend.round(3))                                # por anillo, sin fila conjunta
print(res_sp.placebo[["ring", "label", "effect", "se", "p"]].round(3))
print(res_sp.tests[["test", "stat", "df", "p", "underpowered"]].round(4))
print(round(res_sp.outer_ring_contrast_bound, 4),
      round(res_sp.outer_ring_contrast_share, 3))
fig_sp = res_sp.plot()      # efectos por anillo con el DiD ingenuo en línea discontinua, y el estudio de eventos
```

- **`ring_table`** es la tabla principal: una fila por anillo más una fila final para el grupo de control omitido, con `n_units` (unidades alguna vez en el anillo), `n_obs` (unidad-períodos que llevan la dummy) y `dropped`. Su fila final cuenta las unidades *nunca* expuestas en ningún período (72 aquí), no las 409 que la tabla `counts` de la asignación reporta para el mismo anillo, que cuenta las unidades que estuvieron *alguna vez* en él. Siempre proviene del **ajuste estático de efectos fijos bidireccionales**; `static_estimator` está fijado en `'twfe'` y `event_study_estimator` gobierna únicamente el estudio de eventos. Con más de una cohorte ese ajuste estático arrastra el problema de ponderaciones negativas de Goodman-Bacon (2021) / de Chaisemartin-D'Haultfœuille (2020), una vez por anillo, y `summary()` lo dice — lea `att_by_ring_es` a su lado.
- **`total_effect`** es $\delta_0 + \sum_{r\ge1} (N_r/N_0)\,\delta_r$: el efecto agregado por unidad tratada, incluida su huella de desbordamiento. $N_0$ cuenta las unidades alguna vez tratadas y $N_r$ las nunca tratadas cuyo anillo de *entrada* es $r$, de modo que cada unidad se cuenta exactamente una vez. Con 20 plantas y 317 vecinos expuestos resulta mucho mayor que $\delta_0$; eso es aritmética, no un efecto de tratamiento mayor.
- **`event_study`** indexa cada celda por el anillo **actual** de la unidad con el reloj anclado en la entrada, así que una unidad que migra hacia dentro no arrastra consigo la trayectoria dinámica de su banda anterior. `att_by_ring_es` lo agrega de vuelta a un número por anillo sobre $e \ge 0$ con pesos de conteo de celdas post; no coincide en general con el `effect` estático, porque el coeficiente estático es un promedio ponderado tipo GLS de la trayectoria dinámica y no uno ponderado por conteos.
- **`pretrend`** reporta una fila de Wald por anillo para "todo $\delta_{r,e}$ con $e \le -2$ es cero". **`placebo`** reajusta toda la especificación estática sobre la submuestra previa a la exposición con la fecha de entrada adelantada `placebo_periods` períodos, y sus `n_units` / `n_obs` cuentan las unidades que llevan la dummy falsa de ese anillo, no el tamaño de la muestra placebo.
- **`tests`** contiene `outer_ring`, `no_spillover`, `equal_rings` y una fila `dynamics_ring*` por anillo, cada una con su bandera `underpowered`.

### 6.4 Anillos de contigüidad, y qué ocurre cuando un anillo es delgado

Cuando "a qué distancia" significa "cuántas fronteras", `contiguity_rings` mide los anillos en saltos sobre el grafo de adyacencia en lugar de kilómetros — la versión Berg-Reisinger-Streitz / Delgado-Florax del diseño. Los centroides estatales que incluye el paquete traen una columna `neighbors` con los vecinos por frontera terrestre, que es exactamente la entrada que quiere `contiguity_weights`. Un conteo de saltos no tiene escala en kilómetros para un núcleo de Conley, así que esta ruta fuerza `cov_type="cluster"`:

```python
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import contiguity_weights
from puremacro.did import contiguity_rings

states = load_us_state_centroids()
mainland = states[states["neighbors"].str.strip() != ""]     # descarta AK, HI, PR
W = contiguity_weights({s: nb.split() for s, nb in mainland["neighbors"].items()})

adopt = {"CA": 2016.0, "OR": 2016.0, "WA": 2016.0,
         "NY": 2019.0, "MA": 2019.0, "IL": 2019.0}
adopt_year = pd.Series({s: adopt.get(s, np.nan) for s in W.ids})
yrs = list(range(2012, 2023))
ra_hop = contiguity_rings(W, treat_time=adopt_year, times=yrs, orders=2)
print(ra_hop.summary())

rng_hop = np.random.default_rng(3)
eff = {0: 0.8, 1: 0.3, 2: 0.1, 3: 0.0}
ring_at = ra_hop.frame.set_index(["unit", "time"])["ring"]
mu_s = {s: rng_hop.normal() for s in W.ids}
tau_s = {y: 0.2 * rng_hop.normal() for y in yrs}
panel_hop = pd.DataFrame([
    {"state": s, "year": y, "adopt_year": adopt_year[s],
     "log_wage": mu_s[s] + tau_s[y] + eff[int(ring_at[(s, y)])]
                 + 0.2 * rng_hop.normal()}
    for s in W.ids for y in yrs
])

res_hop = spatial_did(panel_hop, unit="state", time="year", outcome="log_wage",
                      treat_time="adopt_year", assignment=ra_hop,
                      cov_type="cluster")
print(res_hop.summary())
print(res_hop.tests[["test", "stat", "df", "p", "underpowered"]])
```

Pasar `assignment=` fija las bandas, la temporalidad y la métrica, así que `rings=`, `ring_timing=` y `metric=` lanzan un error si además los especifica — el paquete se niega a devolver como eco un argumento que no hizo nada.

Seis estados tratados dejan el anillo 0 con seis unidades, y ahí es donde muerde la barrera de tamaño: la fila `dynamics_ring0` de `tests` y la fila del anillo 0 de `pretrend` vuelven con `stat = p = NaN` y `underpowered=True`, y `summary()` imprime `NOT REPORTED` en lugar de un valor p. Todo estadístico de Wald *conjunto* aquí es una forma cuadrática en una covarianza robusta; cuando un anillo lo sostienen un puñado de unidades, esa covarianza se comporta como una covarianza muestral con esos pocos grados de libertad y la referencia chi-cuadrado es gravemente incorrecta. Medido sobre diseños nulos con semilla a un nominal de 0.10, la pre-tendencia por anillo rechaza 0.285 con menos de diez unidades en el anillo frente a 0.085–0.144 por encima, y `no_spillover` 0.231 frente a 0.079–0.110; diez unidades es donde desaparece la distorsión, y por eso el umbral está ahí. En lugar de imprimir un valor p confiado desde una distribución de referencia que sabe equivocada, `spatial_did` deja la fila en blanco y avisa nombrando los anillos. La tabla de anillos **por coeficiente** no se ve afectada: un Wald de un solo coeficiente sigue estando correctamente dimensionado por pequeño que sea el anillo, y por eso `outer_ring` nunca se bloquea.

### 6.5 Lo que este diseño no puede decirle

Las advertencias siguientes no son cautelas retóricas; cada una señala un punto donde el estimador calla o se equivoca, y todas están en el propio docstring del módulo.

- **La prueba del anillo exterior acota una diferencia, no un nivel.** $\delta_R$ está identificado como (anillo exterior) menos (los controles más allá del anillo). Un desbordamiento *común* a ambos los desplaza por igual y deja $\delta_R$ exactamente en cero. Así que `outer_ring_contrast_bound` = $|\delta_R| + z\,\text{se}_R$ descarta diferencias mayores que él mismo; no dice nada sobre el nivel del desbordamiento allá afuera. El nombre del campo, el docstring y `summary()` dicen "contrast" por esa razón, y la única defensa real es la sensibilidad a ensanchar `rings[-1]`.
- **No hay fila de pre-tendencia conjunta sobre todos los anillos.** Doce restricciones leídas de una covarianza robusta sostenida por unas pocas decenas de unidades tratadas no son una chi-cuadrado: sobre nulos con semilla a un nominal de 0.10 rechazó 0.515 (90 unidades / 9 tratadas), 0.20 y 0.18 (200 / 22, con todos los anillos superando la barrera de tamaño), 0.135 (200 / 20), 0.095 (300 / 30) y 0.087 (400 / 45). Está sobredimensionada justo donde el diseño es lo bastante pequeño como para que la pregunta importe, así que no se publica. `pretrend` lleva la misma información tres restricciones cada vez; un ajuste de Bonferroni o de Holm entre esas filas es la afirmación conjunta honesta. De las filas conjuntas que sí se publican, la pre-tendencia por anillo es la menos fiable (0.10–0.17 a un nominal de 0.10 incluso pasada la barrera): léala como indicativa.
- **No existe un `estimator="cs"` que recorra los anillos con Callaway-Sant'Anna.** Necesitaría un único momento de tratamiento escalar por unidad, así que no podría representar a una unidad que está en el anillo 2 en $t=5$ y en el anillo 1 en $t=8$; no produce covarianza entre anillos, lo que deja indefinidas la descomposición de contaminación y todas las pruebas conjuntas; y su bootstrap de panel ignora exactamente la correlación espacial para la que existe este módulo. Ejecute usted mismo `callaway_santanna` sobre el subpanel de cada anillo si quiere esas estimaciones puntuales.
- **Los puntos de corte de los anillos son un grado de libertad del investigador.** No hay ventana de anticipación, ni diseño de rosquilla (doughnut), ni medida continua de exposición, ni selección automática de fronteras. Fije los cortes ex ante por razones sustantivas y reporte sensibilidad; `RingAssignment.plot()` existe para ayudarle a cortar donde la distribución de distancias es delgada.
- **Los valores críticos son normales estándar, nunca $t$.** Bajo dependencia espacial los grados de libertad efectivos no son $n - k$, así que no se ofrece ninguna distribución $t$.
- **Sin mapas.** El módulo es numpy / scipy / pandas puro para poder ejecutarse bajo Pyodide; los mapas coropléticos y cualquier pila geométrica quedan fuera de alcance, y `matplotlib` se importa dentro de `plot()`.
- **Los anillos pequeños se descartan, nunca se fusionan.** `min_units_per_ring` descarta las *filas* de un anillo poco poblado; nunca las reasigna al grupo de control, lo que contaminaría justamente la comparación de la que depende el diseño. Los anillos descartados se marcan en `ring_table["dropped"]` y se listan en `dropped_rings`.
- **La "carne" del HAC puede ser indefinida.** El núcleo de Bartlett bidimensional no es definido positivo, así que una varianza sándwich puede salir negativa; el SE, el t, el p y el IC de ese coeficiente son entonces `NaN` en lugar de la raíz cuadrada de un número negativo. `psd_adjust="clip"` proyecta la matriz sobre el cono PSD, lo cual es conservador: toda varianza aumenta débilmente.

---

## 7. Salida para publicación

Todos los objetos de resultado de `puremacro.did` (y `puremacro.synthetic_control.SyntheticControlResult`) exponen `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` y `plot()`; los exportadores nunca emiten una columna de índice posicional. Las herramientas de sensibilidad Honest-DiD que consumen estos resultados se documentan en [honest_did.md](honest_did.md).

```python
print(res_sa.to_latex())
print(res_multi.to_typst())
fig_cs = res_cs.plot()
fig_cdh = res_cdh.plot()
fig_multi = res_multi.plot()
```

---

## Resumen de pautas de diagnóstico

1. **Tendencias paralelas pre-tratamiento**: Inspeccione siempre los coeficientes del estudio de eventos para $e < 0$ en `callaway_santanna` / `sun_abraham`; deben ser estadísticamente indistinguibles de cero. `borusyak_jaravel_spiess` solo reporta filas post-tratamiento; use el `placebo_p` de `cdh_did` (placebo de cambiantes) como su contraste de pre-tendencias.
2. **Nunca tratados vs. no tratados aún**:
   - Si existe un grupo genuinamente nunca tratado, fije `control="never_treated"`.
   - Si todas las unidades acaban tratadas, use `control="not_yet_treated"` para no descartar a los últimos adoptantes (`control_group=` se acepta como alias de `control=`). El estimador de imputación BJS no puede identificar los períodos posteriores a la última adopción en esos paneles: lanza un error salvo que se indique `unidentified="drop"`.
3. **Desbordamientos**: Si el tratamiento puede alcanzar a una unidad no tratada cercana, el grupo de control está contaminado y todos los estimadores anteriores estiman el efecto directo *menos* un término de contaminación. Use `spatial_did` y lea `contamination` junto a `direct_effect`; el anillo exterior es la única evidencia de que el grupo de comparación restante está limpio, y acota una diferencia, no un nivel.
4. **Tablas para publicación**: Exporte cualquier objeto de resultado directamente a LaTeX o Typst mediante `.to_latex()` y `.to_typst()`.

## Referencias

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. y Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review* 111(12), 4088–4118.
- Berg, T., Reisinger, M. y Streitz, D. (2021). Spillover effects in empirical corporate finance. *Journal of Financial Economics* 142(3), 1109–1127.
- Borusyak, K., Jaravel, X. y Spiess, J. (2024). Revisiting event-study designs: robust and efficient estimation. *Review of Economic Studies* 91(6), 3253–3285.
- Butts, K. (2023). Difference-in-differences estimation with spatial spillovers. arXiv:2105.03737.
- Callaway, B. y Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics* 225(2), 200–230.
- Clarke, D. (2017). Estimating difference-in-differences in the presence of spillovers. MPRA Paper 81604.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- de Chaisemartin, C. y D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review* 110(9), 2964–2996.
- Delgado, M. S. y Florax, R. J. G. M. (2015). Difference-in-differences techniques for spatial data: local autocorrelation and spatial interaction. *Economics Letters* 137, 123–126.
- Driscoll, J. C. y Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics* 80(4), 549–560.
- Frisch, R. y Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica* 1(4), 387–401.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics* 225(2), 254–277.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
- Huber, M. y Steinmayr, A. (2021). A framework for separating individual-level treatment effects from spillover effects. *Journal of Business & Economic Statistics* 39(2), 422–436.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A. y Poe, J. (2023). What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics* 235(2), 2218–2244.
- Sun, L. y Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics* 225(2), 175–199.
- Verbitsky-Savitz, N. y Raudenbush, S. W. (2012). Causal inference under interference in spatial settings. *Epidemiologic Methods* 1(1), 107–130.
