> 🇬🇧 [English](../did.md) · 🇪🇸 Español

# Diferencias en Diferencias Modernas (DiD)

Las regresiones clásicas con efectos fijos bidireccionales (Two-Way Fixed Effects, TWFE) de la forma:

$$y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$

fallan sistemáticamente cuando el momento de adopción del tratamiento es **escalonado** (diferentes unidades reciben el tratamiento en distintos períodos) y los efectos del tratamiento son **heterogéneos** entre cohortes o dinámicos en el tiempo (Goodman-Bacon 2021, de Chaisemartin y D'Haultfœuille 2020). Bajo efectos heterogéneos, TWFE utiliza implícitamente a unidades ya tratadas como controles para unidades tratadas posteriormente, generando ponderaciones negativas que pueden llegar a invertir el signo de la estimación.

`puremacro.did` implementa la suite completa de estimadores robustos a la heterogeneidad en Python puro, ofreciendo inferencia exacta por bootstrap, agregaciones dinámicas para estudios de eventos y exportación directa de tablas para publicaciones.

---

## Panorama de estimadores

| Estimador | Función | Referencia clave | Estrategia econométrica |
|---|---|---|---|
| **Callaway y Sant'Anna** | `callaway_santanna` | Callaway y Sant'Anna (2021, *J. Econometrics*) | Efectos de grupo y tiempo $ATT(g, t)$ con grupos de control limpios (nunca tratados o no tratados aún) |
| **Sun y Abraham** | `sun_abraham` | Sun y Abraham (2021, *J. Econometrics*) | Estudio de eventos con ponderación por cohortes para aislar cambios composicionales |
| **Borusyak, Jaravel y Spiess** | `borusyak_jaravel_spiess` | Borusyak, Jaravel y Spiess (2024, *Rev. Econ. Stud.*) | Estimador de imputación: ajusta efectos fijos sobre observaciones no tratadas y proyecta contrafactuales |
| **DiD Sintético (SDID)** | `synthetic_did` | Arkhangelsky, Athey et al. (2021, *AER*) | Doble ponderación: pesos de unidad $\omega$ para ajustar pre-tendencias + pesos temporales $\lambda$ |
| **SDID multi-cohorte** | `sdid_multi_cohort` | Roth et al. (2023, *J. Econometrics*) | DiD sintético generalizado a múltiples cohortes de adopción |

---

## 1. Callaway y Sant'Anna (2021)

Estima los efectos medios del tratamiento para cada cohorte $g$ (año de adopción) en cada período $t$, denotados como $ATT(g, t)$, y los agrega en un perfil de estudio de eventos relativo al momento de intervención $e = t - g$:

```python
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

res_cs = callaway_santanna(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control_group="never_treated",  # o "not_yet_treated"
    n_boot=1000,
    ci=0.95,
)

print(res_cs.summary())
print(res_cs.att_event_study.head())
print(res_cs.to_latex())
print(res_cs.to_typst())
```

Atributos principales de `CallawaySantannaResult`:
- `att_gt`: DataFrame con las estimaciones $ATT(g, t)$ y sus errores estándar bootstrap.
- `att_event_study`: Efectos dinámicos agregados según el tiempo transcurrido desde el tratamiento.
- `att_overall`: Efecto medio agregado post-tratamiento.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Métodos de exportación para manuscritos.

---

## 2. Sun y Abraham (2021)

Sun y Abraham modelan explícitamente las trayectorias de cada cohorte y ponderan los coeficientes dinámicos por la participación muestral de cada grupo, garantizando que el perfil dinámico no se contamine por cambios en la composición de las cohortes que identifican cada horizonte:

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
```

---

## 3. Estimador de imputación de Borusyak, Jaravel y Spiess (2024)

El estimador BJS es asintóticamente eficiente bajo el supuesto de tendencias paralelas y opera en tres pasos intuitivos:
1. **Ajuste**: Estima los efectos fijos de unidad y tiempo utilizando *únicamente* las observaciones no tratadas ($D_{it} = 0$).
2. **Imputación**: Proyecta los resultados contrafactuales $\hat{y}_{it}(0)$ para las celdas tratadas.
3. **Promedio**: Calcula el efecto como $\hat{\tau}_{it} = y_{it} - \hat{y}_{it}(0)$ y agrega sobre el horizonte de eventos.

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
```

---

## 4. Diferencias en diferencias sintéticas (SDID)

Arkhangelsky et al. (2021) unifican el control sintético y las diferencias en diferencias:
- A diferencia del control sintético tradicional, SDID es invariante a desplazamientos aditivos de nivel entre unidades y a lo largo del tiempo.
- A diferencia del DiD clásico, no exige tendencias paralelas entre el grupo tratado y todas las unidades de control; en su lugar, optimiza pesos no negativos $\omega_i \ge 0$ para alinear las tendencias previas y pesos temporales $\lambda_t \ge 0$ para ponderar los períodos pre-tratamiento más relevantes:

```python
from puremacro.did import synthetic_did

res_sdid = synthetic_did(
    df,
    unit="state",
    time="quarter",
    outcome="gdp_growth",
    treatment="has_reform",
)
print(res_sdid.summary())
print("Principales unidades donantes:\n", res_sdid.omega[res_sdid.omega > 0.05])
```
