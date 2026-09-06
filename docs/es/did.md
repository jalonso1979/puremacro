> 🇬🇧 [English](../did.md) · 🇪🇸 Español

# Diferencias en Diferencias Modernas (DiD)

Las regresiones clásicas con efectos fijos bidireccionales (Two-Way Fixed Effects, TWFE) de la forma:

$$y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$

fallan sistemáticamente cuando el momento de adopción del tratamiento es **escalonado** (diferentes unidades reciben el tratamiento en distintos períodos) y los efectos del tratamiento son **heterogéneos** entre cohortes o dinámicos en el tiempo (Goodman-Bacon 2021, de Chaisemartin y D'Haultfœuille 2020). Bajo efectos heterogéneos, TWFE utiliza implícitamente a unidades ya tratadas como controles para unidades tratadas posteriormente, generando ponderaciones negativas que pueden llegar a invertir el signo de la estimación.

`puremacro.did` implementa la suite completa de estimadores robustos a la heterogeneidad en Python puro (solo numpy / scipy / pandas), ofreciendo inferencia por bootstrap de panel, agregaciones dinámicas para estudios de eventos y exportación directa de tablas para publicaciones.

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

Conviven dos convenciones de entrada. `callaway_santanna`, `sun_abraham`, `borusyak_jaravel_spiess` y `synthetic_did` reciben un **DataFrame en formato largo** con los nombres de columna en `unit=`, `time=`, `outcome=`, `treat_time=` (`treat_time` es el primer período de tratamiento de cada unidad, `NaN` para las nunca tratadas). `cdh_did` y `sdid_multi_cohort` reciben **cuatro arrays 1-D alineados** `(y, treatment, panel_id, time_id)`, donde `treatment` es el estado de tratamiento 0/1 de cada fila.

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

Atributos principales de `CallawaySantannaResult`:
- `att_gt`: DataFrame con las estimaciones $ATT(g, t)$ y sus errores estándar bootstrap (columnas `g, t, event_time, att, se, lo, hi`).
- `att_event_study`: Efectos dinámicos agregados según el tiempo transcurrido desde el tratamiento; cada horizonte es la media sin ponderar de las cohortes que lo identifican (`n_cohorts`).
- `att_overall`: Media simple de las celdas post-tratamiento $ATT(g, t)$ (cada celda cohorte-período identificada cuenta una vez; *no* se pondera por tamaño de cohorte — use `sun_abraham` para un efecto global ponderado por participación de unidades).
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Métodos de exportación del estudio de eventos (sin columna de índice); `.plot()` lo dibuja con su banda de confianza.

---

## 2. Sun y Abraham (2021)

Sun y Abraham modelan explícitamente las trayectorias de cada cohorte y ponderan los coeficientes dinámicos por la participación muestral de cada grupo, garantizando que el perfil dinámico no se contamine por cambios en la composición de las cohortes que identifican cada horizonte. `att_overall` es la media de los $ATT(g, t)$ post-tratamiento ponderada por la participación de cada cohorte:

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
- A diferencia del control sintético tradicional, SDID es invariante a desplazamientos aditivos de nivel entre unidades y a lo largo del tiempo: ambos problemas de pesos incluyen los interceptos $\omega_0$, $\lambda_0$ del artículo, de modo que sumar una constante a la trayectoria de cualquier unidad (o una constante común a cualquier período) no altera $\hat\tau$.
- A diferencia del DiD clásico, no exige tendencias paralelas entre el grupo tratado y todas las unidades de control; en su lugar, optimiza pesos no negativos $\omega_i \ge 0$ para alinear las tendencias previas y pesos temporales $\lambda_t \ge 0$ para ponderar los períodos pre-tratamiento más relevantes:

$$\hat{\tau}^{\text{SDID}} = \arg\min_{\tau, \mu, \alpha, \beta} \sum_{i=1}^N \sum_{t=1}^T \left( y_{it} - \mu - \alpha_i - \beta_t - \tau W_{it} \right)^2 \hat{\omega}_i \hat{\lambda}_t$$

`synthetic_did` maneja una **única cohorte de tratamiento** (un período de adopción común, `treat_time` igual para todas las unidades tratadas y `NaN` para los donantes) y requiere un **panel balanceado**: una celda `(unit, time)` ausente lanza un `ValueError` que la identifica. Los errores estándar provienen de un bootstrap sobre donantes.

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

Para adopción escalonada en varias cohortes, use `sdid_multi_cohort`. Ejecuta `synthetic_did` una vez por cohorte de adopción y promedia las estimaciones con pesos por tamaño de cohorte. El grupo de donantes de cada cohorte permanece no tratado durante toda su ventana SDID: `control="never_treated"` usa las unidades nunca tratadas sobre el panel completo, `control="not_yet_treated"` admite también unidades tratadas más tarde pero trunca la ventana en su primera fecha de adopción, y el valor por defecto `"auto"` elige donantes nunca tratados cuando hay al menos dos. Como `cdh_did`, usa la forma de cuatro arrays:

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

## 6. Salida para publicación

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
3. **Tablas para publicación**: Exporte cualquier objeto de resultado directamente a LaTeX o Typst mediante `.to_latex()` y `.to_typst()`.
